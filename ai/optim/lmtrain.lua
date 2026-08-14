local Tensor = require("tensor2")
local Transformer = require("transformer")
local optim2 = require("optim2")
local matrix = require("matrix")

local LMTrain = {}
LMTrain.__index = LMTrain

local floor, ceil = math.floor, math.ceil
local huge = math.huge
local exp, sqrt, cos, pi = math.exp, math.sqrt, math.cos, math.pi
local sformat = string.format
local srep = string.rep

local DEFAULTS = {
vocab = 256, dim = 64, layers = 2, heads = 4, ffn = 128,
epochs = 200, lr = 1e-3, lr_min = 0, warmup = 0.05, sched = "cosine",
optimizer = "adamw", weight_decay = 0.01, beta1 = 0.9, beta2 = 0.95,
momentum = 0.9,
bar_style = 1, bar_len = 20, every = 1,
patience = 0, min_delta = 0,
verbose = true, causal = true, pos = "rope", rope_base = 10000,
max_seq = 128, stride = nil, batch_size = 8, shuffle = true,
val_split = 0, eval_every = 1, keep_best = true,
clip_norm = 1.0, dropout = 0, tie = true,
}

local NILABLE = {
data = true, log = true, stop_loss = true, stop_when = true,
seed = true, on_stop = true, name = true, tokenizer = true,
temperature = true, top_k = true, top_p = true,
}

local ARCH_KEYS = {
vocab = true, dim = true, layers = true, heads = true, ffn = true,
causal = true, pos = true, rope_base = true, max_seq = true,
tie = true, dropout = true,
}

local OPT_KEYS = {
optimizer = true, weight_decay = true, beta1 = true, beta2 = true, momentum = true,
}

local RESERVED = {
new = true, run = true, config = true, bar = true, reset = true,
summary = true, parameters = true, generate = true, save = true,
load = true, push = true, encode = true, decode = true, evaluate = true,
model = true, params = true, opt = true, sgd = true, history = true,
epoch = true, loss = true, best_loss = true, accuracy = true,
elapsed = true, stopped = true, val_loss = true,
}

local ALIASES = {
e = "epochs", epoch = "epochs", epochs = "epochs", n = "epochs",
iters = "epochs", steps = "epochs",
lr = "lr", rate = "lr", learning_rate = "lr",
lr_min = "lr_min", min_lr = "lr_min",
warmup = "warmup", warmup_frac = "warmup",
sched = "sched", schedule = "sched", lr_decay = "sched",
opt = "optimizer", optimizer = "optimizer", optim = "optimizer",
wd = "weight_decay", weight_decay = "weight_decay", decay = "weight_decay",
beta1 = "beta1", beta2 = "beta2", momentum = "momentum",
v = "vocab", vocab = "vocab", vocab_size = "vocab",
d = "dim", dim = "dim", width = "dim", model_dim = "dim",
l = "layers", layers = "layers", depth = "layers", nlayers = "layers",
h = "heads", heads = "heads", nheads = "heads",
f = "ffn", ffn = "ffn", ff = "ffn", hidden = "ffn",
data = "data", tokens = "data", seq = "data", sequences = "data",
tokenizer = "tokenizer", tok = "tokenizer",
bar = "bar_style", style = "bar_style", bar_style = "bar_style",
bar_len = "bar_len", width_bar = "bar_len",
log = "log", on_epoch = "log", callback = "log",
every = "every", log_every = "every",
stop = "stop_loss", stop_loss = "stop_loss", target_loss = "stop_loss",
stop_when = "stop_when", ["until"] = "stop_when",
patience = "patience", min_delta = "min_delta",
verbose = "verbose", seed = "seed", name = "name", on_stop = "on_stop",
causal = "causal", mask = "causal", causal_mask = "causal",
pos = "pos", positional = "pos", position = "pos", pe = "pos",
rope_base = "rope_base", theta = "rope_base",
max_seq = "max_seq", ctx = "max_seq", max_ctx = "max_seq", context = "max_seq",
stride = "stride", batch = "batch_size", batch_size = "batch_size", bs = "batch_size",
shuffle = "shuffle", val = "val_split", val_split = "val_split",
eval_every = "eval_every", keep_best = "keep_best",
clip = "clip_norm", clip_norm = "clip_norm", grad_clip = "clip_norm",
dropout = "dropout", drop = "dropout", tie = "tie",
temperature = "temperature", temp = "temperature",
top_k = "top_k", topk = "top_k", top_p = "top_p", topp = "top_p",
preset = "preset",
}

local BAR_CHARS = {
[1] = { fill = "#", empty = "-" },
[2] = { fill = "\226\150\147", empty = "\226\150\145" },
[3] = { fill = "=", empty = " " },
}

local POS_MODES = { rope = true, rotary = true, sinusoidal = true, sin = true,
                absolute = true, abs = true, none = true, off = true }
local SCHEDULES = { cosine = true, linear = true, none = true, constant = true }

local function is_int(x)
return type(x) == "number" and x == floor(x) and x == x and x ~= huge and x ~= -huge
end

local function is_finite(x)
return type(x) == "number" and x == x and x ~= huge and x ~= -huge
end

function LMTrain.encode(text)
if type(text) ~= "string" then
    error("LMTrain.encode: expected a string, got " .. type(text), 2)
end
local t = {}
for i = 1, #text do t[i] = string.byte(text, i) + 1 end
return t
end

function LMTrain.decode(tokens)
local chars = {}
for i = 1, #tokens do
    local id = tokens[i]
    if type(id) ~= "number" or id < 1 or id > 256 then
        error(sformat("LMTrain.decode: token %d = %s cannot be a byte (need 1..256); "
            .. "this model's vocab is larger than a byte alphabet", i, tostring(id)), 2)
    end
    chars[i] = string.char(id - 1)
end
return table.concat(chars)
end

local function coerce_data(config)
local out = {}
for k, v in pairs(config) do out[k] = v end

local function encode_text(text)
    if out.tokenizer then return out.tokenizer:encode(text) end
    return LMTrain.encode(text)
end

if type(out.data) == "string" then
    out.data = encode_text(out.data)
    if out.vocab == nil then
        out.vocab = out.tokenizer and out.tokenizer.vocab_size or 256
    end
elseif type(out.data) == "table" and type(out.data[1]) == "string" then
    local seqs = {}
    for i = 1, #out.data do seqs[i] = encode_text(out.data[i]) end
    out.data = seqs
    if out.vocab == nil then
        out.vocab = out.tokenizer and out.tokenizer.vocab_size or 256
    end
end
return out
end

local function resolve_aliases(config, who)
if type(config) ~= "table" then
    error(sformat("%s: expected a config table, got %s", who, type(config)), 3)
end
local resolved, origin = {}, {}
for key, val in pairs(config) do
    if type(key) ~= "string" then
        error(sformat("%s: config keys must be strings, got a %s key", who, type(key)), 3)
    end
    local canonical = ALIASES[key] or key
    if RESERVED[canonical] then
        error(sformat("%s: '%s' is a reserved name and cannot be used as a config key",
            who, key), 3)
    end
    local prev = origin[canonical]
    if prev ~= nil and prev ~= key then
        error(sformat("%s: conflicting config keys '%s' and '%s' both map to '%s', pick one",
            who, prev, key, canonical), 3)
    end
    resolved[canonical] = val
    origin[canonical] = key
end
if resolved.sched == true then resolved.sched = "cosine" end
if resolved.sched == false then resolved.sched = "none" end
return resolved
end

local function get_sequences(data, who)
if type(data) ~= "table" then
    error(sformat("LMTrain%s: data must be a table of token ids, got %s", who, type(data)), 3)
end
if #data == 0 then
    error(sformat("LMTrain%s: data is empty, expected at least 2 token ids", who, 3), 3)
end
if type(data[1]) == "table" then return data end
return { data }
end

local function scan_tokens(data)
if type(data) ~= "table" or #data == 0 then return nil end
local sequences = (type(data[1]) == "table") and data or { data }
local max_id = 0
for s = 1, #sequences do
    local seq = sequences[s]
    if type(seq) == "table" then
        for i = 1, #seq do
            local tok = seq[i]
            if is_int(tok) and tok > max_id then max_id = tok end
        end
    end
end
if max_id < 2 then return nil end
return max_id
end

local function validate_data(data, vocab, who)
local sequences = get_sequences(data, who)
for s = 1, #sequences do
    local seq = sequences[s]
    if type(seq) ~= "table" then
        error(sformat("LMTrain%s: sequence %d must be a table, got %s", who, s, type(seq)), 3)
    end
    local len = #seq
    if len < 2 then
        error(sformat("LMTrain%s: sequence %d has %d token(s) but next-token training needs at least 2",
            who, s, len), 3)
    end
    for i = 1, len do
        local tok = seq[i]
        if not is_int(tok) then
            error(sformat("LMTrain%s: sequence %d, token %d must be an integer id, got %s",
                who, s, i, tostring(tok)), 3)
        end
        if tok < 1 or tok > vocab then
            error(sformat("LMTrain%s: sequence %d, token %d = %d is outside 1..%d, raise 'vocab' or remap your tokens",
                who, s, i, tok, vocab), 3)
        end
    end
end
return sequences
end

local function build_windows(sequences, max_seq, stride)
stride = stride or max_seq
local windows = {}
for s = 1, #sequences do
    local seq = sequences[s]
    local len = #seq
    local start = 1
    while start < len do
        local stop = start + max_seq
        if stop > len then stop = len end
        local inp, tgt = {}, {}
        local n = 0
        for i = start, stop - 1 do
            n = n + 1
            inp[n] = seq[i]
            tgt[n] = seq[i + 1]
        end
        if n >= 1 then
            windows[#windows + 1] = { inp, tgt, n }
        end
        if stop >= len then break end
        start = start + stride
    end
end
return windows
end

local function check_config(self, who)
        local function pos_int(key)
            local v = self[key]
            if not is_int(v) or v < 1 then
                error(sformat("LMTrain%s: '%s' must be a positive integer, got %s", who, key, tostring(v)), 4)
            end
        end

        pos_int("vocab"); pos_int("dim"); pos_int("layers")
        pos_int("heads"); pos_int("ffn"); pos_int("epochs")
        pos_int("max_seq"); pos_int("batch_size")

        -- pos_int("max_seq") only ensures it is >= 1.
        -- Next-token prediction needs at least 2 tokens to work.
        if self.max_seq < 2 then
            error(sformat("LMTrain%s: 'max_seq' must be at least 2 for next-token prediction, got %d", who, self.max_seq), 3)
        end

        if self.dim % self.heads ~= 0 then
            error(sformat("LMTrain%s: dim (%d) must be divisible by heads (%d)", who, self.dim, self.heads), 3)
        end
    if type(self.pos) ~= "string" or not POS_MODES[string.lower(self.pos)] then
        error(sformat("LMTrain%s: 'pos' must be 'rope', 'sinusoidal', 'absolute' or 'none', got %s",
            who, tostring(self.pos)), 3)
    end
    local pl = string.lower(self.pos)
    if (pl == "rope" or pl == "rotary") and floor(self.dim / self.heads) < 2 then
        error(sformat("LMTrain%s: RoPE needs dim/heads >= 2 (got dim=%d heads=%d)", who, self.dim, self.heads), 3)
    end
    if type(self.sched) ~= "string" or not SCHEDULES[string.lower(self.sched)] then
        error(sformat("LMTrain%s: 'sched' must be 'cosine', 'linear' or 'none', got %s",
            who, tostring(self.sched)), 3)
    end
    if type(self.causal) ~= "boolean" then
        error(sformat("LMTrain%s: 'causal' must be true or false, got %s", who, tostring(self.causal)), 3)
    end
    if not is_finite(self.lr) or self.lr <= 0 then
        error(sformat("LMTrain%s: 'lr' must be a positive finite number, got %s", who, tostring(self.lr)), 3) 
    end
    if not is_finite(self.lr_min) or self.lr_min < 0 then 
        error(sformat("LMTrain%s: 'lr_min' must be a non-negative number, got %s", who, tostring(self.lr_min)), 3) 
    end
    if self.lr_min > self.lr then 
        error(sformat("LMTrain%s: 'lr_min' (%g) cannot exceed 'lr' (%g)", who, self.lr_min, self.lr), 3) 
    end
    if not is_finite(self.warmup) or self.warmup < 0 or self.warmup >= 1 then 
        error(sformat("LMTrain%s: 'warmup' must be a fraction in [0,1), got %s", who, tostring(self.warmup)), 3) 
    end
    if not is_finite(self.val_split) or self.val_split < 0 or self.val_split >= 1 then 
        error(sformat("LMTrain%s: 'val_split' must be a fraction in [0,1), got %s", who, tostring(self.val_split)), 3) 
    end
    if not is_finite(self.dropout) or self.dropout < 0 or self.dropout >= 1 then 
        error(sformat("LMTrain%s: 'dropout' must be in [0,1), got %s", who, tostring(self.dropout)), 3) 
    end
    if self.stride ~= nil and (not is_int(self.stride) or self.stride < 1) then 
        error(sformat("LMTrain%s: 'stride' must be a positive integer, got %s", who, tostring(self.stride)), 3) 
    end
    if not is_int(self.every) or self.every < 1 then 
        error(sformat("LMTrain%s: 'every' must be a positive integer, got %s", who, tostring(self.every)), 3) 
    end
    if not is_int(self.eval_every) or self.eval_every < 1 then 
        error(sformat("LMTrain%s: 'eval_every' must be a positive integer, got %s", who, tostring(self.eval_every)), 3) 
    end
    if not is_int(self.bar_len) or self.bar_len < 1 then 
        error(sformat("LMTrain%s: 'bar_len' must be a positive integer, got %s", who, tostring(self.bar_len)), 3) 
    end
    if not is_int(self.patience) or self.patience < 0 then 
        error(sformat("LMTrain%s: 'patience' must be a non-negative integer, got %s", who, tostring(self.patience)), 3) 
    end
    if not is_finite(self.min_delta) or self.min_delta < 0 then 
        error(sformat("LMTrain%s: 'min_delta' must be a non-negative number, got %s", who, tostring(self.min_delta)), 3) 
    end
    for _, key in ipairs({ "log", "stop_when", "on_stop" }) do 
        if self[key] ~= nil and type(self[key]) ~= "function" then 
            error(sformat("LMTrain%s: '%s' must be a function, got %s", who, key, type(self[key])), 3) 
        end 
    end
    if self.stop_loss ~= nil and not is_finite(self.stop_loss) then 
        error(sformat("LMTrain%s: 'stop_loss' must be a finite number, got %s", who, tostring(self.stop_loss)), 3) 
    end
    local style = self.bar_style 
    if type(style) == "table" then 
        if type(style.fill) ~= "string" or type(style.empty) ~= "string" then 
            error(sformat("LMTrain%s: custom bar style needs string 'fill' and 'empty' fields", who), 3) 
        end 
    elseif BAR_CHARS[style] == nil then 
        error(sformat("LMTrain%s: unknown bar style %s (expected 1, 2, 3 or { fill = , empty = })", who, tostring(style)), 3) 
    end 
end

local function fmt_time(s)
if s < 60 then return sformat("%4.1fs", s) end
if s < 3600 then return sformat("%dm%02ds", floor(s / 60), floor(s % 60)) end
return sformat("%dh%02dm", floor(s / 3600), floor((s % 3600) / 60))
end

local function default_log(t)
    local style = t.bar_style
    if type(style) ~= "table" then style = BAR_CHARS[style] or BAR_CHARS[1] end
    local total = t.epochs
    local bar_len = t.bar_len or 20
    local filled = 0
    if total and total > 0 then filled = floor((t.epoch / total) * bar_len) end
    if filled < 0 then filled = 0 end
    if filled > bar_len then filled = bar_len end
    local bar_str = srep(style.fill, filled) .. srep(style.empty, bar_len - filled)

    local ppl = t.loss and exp(t.loss < 30 and t.loss or 30) or 0
    local eta = 0
    if t.epoch > 0 and total and total > t.epoch then
        eta = (t.elapsed / t.epoch) * (total - t.epoch)
    end
    local line = sformat("[%s] %d/%d | loss %.4f | ppl %8.2f | acc %5.1f%% | lr %.2e | %s tok/s | eta %s%s",
        bar_str, t.epoch, total, t.loss or 0, ppl, t.accuracy or 0, t.cur_lr or 0,
        sformat("%7.0f", t.tokens_per_sec or 0), fmt_time(eta), t.is_best and " *" or "")
    if t.val_loss then
        line = line .. sformat(" | val %.4f", t.val_loss)
    end
    if t.inplace_bar and t.epoch < (total or 0) then
        io.write("\r", line)
        io.flush()
    else
        io.write("\r", line, "\n")
        io.flush()
    end
end

local function snapshot(params)
local snap = {}
for i = 1, #params do
    local d = params[i].data.data
    local n = params[i].data.rows * params[i].data.cols
    local t = {}
    for k = 1, n do t[k] = d[k] end
    snap[i] = t
end
return snap
end

local function restore(params, snap)
if not snap then return false end
for i = 1, #params do
    local d = params[i].data.data
    local t = snap[i]
    for k = 1, #t do d[k] = t[k] end
end
return true
end

function LMTrain:_build()
if self.seed ~= nil then
    if not is_int(self.seed) then
        error("LMTrain: 'seed' must be an integer, got " .. tostring(self.seed), 2)
    end
    math.randomseed(self.seed)
end
check_config(self, "")
local ok, model = pcall(Transformer.new, {
    vocab = self.vocab, dim = self.dim, layers = self.layers,
    heads = self.heads, ffn = self.ffn, causal = self.causal,
    pos = self.pos, rope_base = self.rope_base, max_seq = self.max_seq,
    tie = self.tie, dropout = self.dropout,
})
if not ok then
    error(sformat("LMTrain: failed to build Transformer{ vocab=%d, dim=%d, layers=%d, heads=%d, ffn=%d, pos=%s, causal=%s }: %s",
        self.vocab, self.dim, self.layers, self.heads, self.ffn, tostring(self.pos), tostring(self.causal), tostring(model)), 2)
end
if type(model) ~= "table" or type(model.forward) ~= "function" or type(model.parameters) ~= "function" then
    error("LMTrain: Transformer.new did not return a usable model (needs :forward and :parameters)", 2)
end
local ok_p, params = pcall(model.parameters, model)
if not ok_p or type(params) ~= "table" or #params == 0 then
    error("LMTrain: model:parameters() failed or returned nothing to optimize: " .. tostring(params), 2)
end
local ok_o, opt = pcall(optim2.create, self.optimizer, params, {
    lr = self.lr, weight_decay = self.weight_decay,
    beta1 = self.beta1, beta2 = self.beta2, momentum = self.momentum,
})
if not ok_o or type(opt) ~= "table" then
    error("LMTrain: optimizer construction failed: " .. tostring(opt), 2)
end
if type(opt.zero_grad) ~= "function" or type(opt.step) ~= "function" then
    error("LMTrain: optimizer must expose zero_grad() and step()", 2)
end
self.model = model
self.params = params
self.opt = opt
self.sgd = opt
self.base_lr = self.lr
self.best_snapshot = nil
self._dirty = false
return self
end

function LMTrain.new(config)
config = resolve_aliases(config or {}, "LMTrain")
config = coerce_data(config)
local self = setmetatable({}, LMTrain)

self.history = {}
self.best_loss = huge
self.epoch = 0
self.loss = nil
self.val_loss = nil
self.is_best = false
self.stopped = nil
self.elapsed = 0
self.tokens_per_sec = 0
self.inplace_bar = true

local preset = config.preset
config.preset = nil

for key, val in pairs(config) do self[key] = val end
for key, default_val in pairs(DEFAULTS) do
    if self[key] == nil then self[key] = default_val end
end

self._auto_vocab = (config.vocab == nil)
if self._auto_vocab and self.data ~= nil then
    local max_id = scan_tokens(self.data)
    if max_id and max_id > self.vocab then self.vocab = max_id end
end

if preset == "auto" and self.data then
    local n = (type(self.data[1]) == "table") and #self.data[1] or #self.data
    local d, l, h, f
    if n < 100 then d, l, h, f = 32, 1, 4, 64
    elseif n < 1000 then d, l, h, f = 64, 2, 4, 128
    else d, l, h, f = 128, 4, 8, 256 end
    if config.dim == nil then self.dim = d end
    if config.layers == nil then self.layers = l end
    if config.heads == nil then self.heads = h end
    if config.ffn == nil then self.ffn = f end
end

self.base_lr = self.lr
self:_build()
return self
end

function LMTrain:config(opts)
opts = resolve_aliases(opts or {}, "LMTrain:config")
if opts.tokenizer == nil and self.tokenizer ~= nil then opts.tokenizer = self.tokenizer end
opts = coerce_data(opts)
local rebuild = false
for key, val in pairs(opts) do
    self[key] = val
    if ARCH_KEYS[key] or OPT_KEYS[key] then rebuild = true end
end
if opts.vocab ~= nil then self._auto_vocab = false end
if opts.data ~= nil and self._auto_vocab then
    local max_id = scan_tokens(self.data)
    if max_id and max_id > self.vocab then
        self.vocab = max_id
        rebuild = true
    end
end
if opts.lr ~= nil then
    if not is_finite(self.lr) or self.lr <= 0 then
        error("LMTrain:config: 'lr' must be a positive finite number, got " .. tostring(self.lr), 2)
    end
    self.base_lr = self.lr
    if self.opt then self.opt.lr = self.lr end
end
if rebuild then self._dirty = true end
return self
end

function LMTrain:bar(opts)
if type(opts) == "number" or type(opts) == "string" then opts = { type = opts } end
opts = opts or {}
if type(opts) ~= "table" then error("LMTrain:bar: expected a style id or an options table", 2) end
local style = opts.type or opts.style or opts.bar
if style ~= nil then
    if type(style) ~= "table" and BAR_CHARS[style] == nil then
        error(sformat("LMTrain:bar: unknown bar type %s", tostring(style)), 2)
    end
    self.bar_style = style
end
local len = opts.len or opts.width or opts.bar_len
if len ~= nil then
    if not is_int(len) or len < 1 then
        error("LMTrain:bar: bar width must be a positive integer, got " .. tostring(len), 2)
    end
    self.bar_len = len
end
if opts.every ~= nil then
    if not is_int(opts.every) or opts.every < 1 then
        error("LMTrain:bar: 'every' must be a positive integer, got " .. tostring(opts.every), 2)
    end
    self.every = opts.every
end
if opts.inplace ~= nil then self.inplace_bar = opts.inplace and true or false end
if opts.off == true then self.verbose = false end
if opts.on == true then self.verbose = true end
return self
end

function LMTrain:reset()
self.history = {}
self.best_loss = huge
self.loss = nil
self.val_loss = nil
self.epoch = 0
self.is_best = false
self.stopped = nil
self.elapsed = 0
self.best_snapshot = nil
self._dirty = true
return self
end

function LMTrain:summary()
return sformat(
    "LMTrain%s: vocab=%d dim=%d layers=%d heads=%d ffn=%d pos=%s causal=%s ctx=%d | %s lr=%g sched=%s bs=%d | epoch=%d loss=%s best=%s",
    self.name and (" " .. tostring(self.name)) or "",
    self.vocab, self.dim, self.layers, self.heads, self.ffn,
    tostring(self.pos), tostring(self.causal), self.max_seq,
    tostring(self.optimizer), self.base_lr or self.lr, tostring(self.sched),
    self.batch_size, self.epoch,
    self.loss and sformat("%.5f", self.loss) or "n/a",
    self.best_loss < huge and sformat("%.5f", self.best_loss) or "n/a")
end

function LMTrain:_lr_at(step, total_steps)
local sched = string.lower(self.sched or "cosine")
if sched == "none" or sched == "constant" then return self.base_lr end
local warm = floor(self.warmup * total_steps)
if warm > 0 and step <= warm then
    return self.base_lr * (step / warm)
end
local denom = total_steps - warm
local progress = denom > 0 and ((step - warm) / denom) or 1
if progress > 1 then progress = 1 end
local lr
if sched == "linear" then
    lr = self.base_lr * (1 - progress)
else
    lr = self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (1 + cos(pi * progress))
end
if lr < self.lr_min then lr = self.lr_min end
return lr
end

function LMTrain:evaluate(windows)
local model = self.model
local prev_training = Tensor.training
Tensor.training = false
local total, count, correct = 0, 0, 0
Tensor.no_grad(function()
    for i = 1, #windows do
        local w = windows[i]
        local logits = model:forward(w[1])
        local loss = Tensor.cross_entropy(logits, w[2])
        total = total + loss:item() * w[3]
        count = count + w[3]
        local ld, lcols = logits.data.data, logits.data.cols
        for r = 1, logits.data.rows do
            local base = (r - 1) * lcols
            local best_id, best_val = 1, -huge
            for c = 1, lcols do
                local v = ld[base + c]
                if v > best_val then best_val, best_id = v, c end
            end
            if best_id == w[2][r] then correct = correct + 1 end
        end
    end
end)
Tensor.training = prev_training
if count == 0 then return 0, 0 end
return total / count, 100 * correct / count
end

function LMTrain:run()
if self.data == nil then
    error("LMTrain: no data set, use LMTrain{ data = {...} } or train:config{ data = {...} }", 2)
end
check_config(self, ":run")
local sequences = validate_data(self.data, self.vocab, ":run")

if self.causal == false and self.verbose then
    print("LMTrain: warning -- causal = false with next-token training lets every "
        .. "position attend to its own target. Loss will drop unrealistically fast.")
end

if self._dirty or self.model == nil or self.opt == nil then self:_build() end

local all = build_windows(sequences, self.max_seq, self.stride)
if #all == 0 then
    error("LMTrain: data produced no training windows", 2)
end

local train_w, val_w = all, {}
if self.val_split > 0 and #all > 1 then
    local nval = floor(#all * self.val_split)
    if nval < 1 then nval = 1 end
    if nval >= #all then nval = #all - 1 end
    train_w, val_w = {}, {}
    for i = 1, #all - nval do train_w[i] = all[i] end
    for i = #all - nval + 1, #all do val_w[#val_w + 1] = all[i] end
end

local model, opt, params = self.model, self.opt, self.params
local nwin = #train_w
local bs = self.batch_size
if bs > nwin then bs = nwin end
local nbatch = ceil(nwin / bs)
local total_steps = self.epochs * nbatch
local order = {}
for i = 1, nwin do order[i] = i end

local started = os.clock()
local stalled = 0
local gstep = 0
self.stopped = nil

for epoch = 1, self.epochs do
    self.epoch = epoch
    Tensor.training = true

    if self.shuffle then
        for i = nwin, 2, -1 do
            local j = math.random(i)
            order[i], order[j] = order[j], order[i]
        end
    end

    local sum_loss, sum_tok, correct, skipped = 0, 0, 0, 0
    local last_norm = 0

    for b = 1, nbatch do
        local lo = (b - 1) * bs + 1
        local hi = lo + bs - 1
        if hi > nwin then hi = nwin end
        local n_in_batch = hi - lo + 1

        opt.zero_grad()
        local batch_tokens = 0
        local batch_loss = 0

        for k = lo, hi do
            local w = train_w[order[k]]
            local logits = model:forward(w[1])
            local loss = Tensor.cross_entropy(logits, w[2])
            loss:backward()
            batch_loss = batch_loss + loss:item() * w[3]
            batch_tokens = batch_tokens + w[3]
            local ld, lcols = logits.data.data, logits.data.cols
            for r = 1, logits.data.rows do
                local base = (r - 1) * lcols
                local best_id, best_val = 1, -huge
                for c = 1, lcols do
                    local v = ld[base + c]
                    if v > best_val then best_val, best_id = v, c end
                end
                if best_id == w[2][r] then correct = correct + 1 end
            end
        end

        if n_in_batch > 1 then
            optim2.scale_grads(params, 1 / n_in_batch)
        end

        local norm = optim2.grad_norm(params)
        last_norm = norm
        local finite = (norm == norm) and norm ~= huge and norm ~= -huge
        if not finite then
            skipped = skipped + 1
        else
            if self.clip_norm and self.clip_norm > 0 and norm > self.clip_norm then
                optim2.scale_grads(params, self.clip_norm / norm)
            end
            gstep = gstep + 1
            opt.lr = self:_lr_at(gstep, total_steps)
            self.cur_lr = opt.lr
            opt.step()
        end

        sum_loss = sum_loss + batch_loss
        sum_tok = sum_tok + batch_tokens
    end

    Tensor.training = false
    local avg_loss = sum_tok > 0 and (sum_loss / sum_tok) or huge
    self.loss = avg_loss
    self.accuracy = sum_tok > 0 and (100 * correct / sum_tok) or 0
    self.grad_norm = last_norm
    self.elapsed = os.clock() - started
    self.tokens_per_sec = self.elapsed > 0 and (sum_tok * epoch / self.elapsed) or 0

    if #val_w > 0 and (epoch % self.eval_every == 0 or epoch == self.epochs) then
        self.val_loss, self.val_accuracy = self:evaluate(val_w)
    end

    local monitor = self.val_loss or avg_loss

    self.history[#self.history + 1] = {
        epoch = epoch, loss = avg_loss, val_loss = self.val_loss,
        accuracy = self.accuracy, lr = self.cur_lr, grad_norm = last_norm,
        time = self.elapsed, skipped = skipped,
    }

    if not is_finite(avg_loss) then
        self.stopped = "diverged"
        self.is_best = false
        if self.verbose then
            io.write("\n")
            print(sformat("LMTrain: stopping at epoch %d, loss became %s. Weights were not updated with non-finite gradients; try a smaller lr.",
                epoch, tostring(avg_loss)))
        end
        break
    end

    self.is_best = monitor < (self.best_loss - self.min_delta)
    if self.is_best then
        self.best_loss = monitor
        stalled = 0
        if self.keep_best then self.best_snapshot = snapshot(params) end
    else
        if monitor < self.best_loss then self.best_loss = monitor end
        stalled = stalled + 1
    end

    if self.verbose and (epoch % self.every == 0 or epoch == self.epochs or epoch == 1) then
        if type(self.log) == "function" then
            local ok, err = pcall(self.log, self)
            if not ok then
                error(sformat("LMTrain: custom log function failed at epoch %d: %s", epoch, tostring(err)), 2)
            end
        else
            default_log(self)
        end
    end

    if self.stop_loss and monitor <= self.stop_loss then
        self.stopped = "stop_loss"
        if self.verbose then
            io.write("\n")
            print(sformat("Stopped early at epoch %d: loss %.5f reached stop_loss %.5f",
                epoch, monitor, self.stop_loss))
        end
        break
    end

    if self.stop_when then
        local ok, hit = pcall(self.stop_when, self)
        if not ok then
            error(sformat("LMTrain: stop_when failed at epoch %d: %s", epoch, tostring(hit)), 2)
        end
        if hit then
            self.stopped = "stop_when"
            if self.verbose then
                io.write("\n")
                print(sformat("Stopped early at epoch %d: stop_when condition met", epoch))
            end
            break
        end
    end

    if self.patience > 0 and stalled >= self.patience then
        self.stopped = "patience"
        if self.verbose then
            io.write("\n")
            print(sformat("Stopped early at epoch %d: no improvement for %d epochs (best %.5f)",
                epoch, stalled, self.best_loss))
        end
        break
    end
end

if self.stopped == nil then self.stopped = "epochs" end
if self.verbose and self.inplace_bar then io.write("\n") end

if self.keep_best and self.best_snapshot and self.stopped ~= "epochs" then
    if restore(self.params, self.best_snapshot) and self.verbose then
        print(sformat("Restored best weights (loss %.5f)", self.best_loss))
    end
end

opt.lr = self.base_lr
self.lr = self.base_lr
Tensor.training = false

if type(self.on_stop) == "function" then pcall(self.on_stop, self) end
return self.model
end

local function pick_token(row, cols, temperature, top_k, top_p)
if not temperature or temperature <= 0 then
    local best_id, best_val = 1, -huge
    for j = 1, cols do
        local v = row[j]
        if v > best_val then best_val, best_id = v, j end
    end
    return best_id
end

top_k = top_k or 0
top_p = top_p or 0

local idx, n
if top_p > 0 and top_p < 1 then
    idx = {}
    for j = 1, cols do idx[j] = j end
    table.sort(idx, function(x, y) return row[x] > row[y] end)
    n = (top_k > 0 and top_k < cols) and top_k or cols
elseif top_k > 0 and top_k < cols then
    idx = {}
    local vals = {}
    for r = 1, top_k do idx[r] = 0; vals[r] = -huge end
    for j = 1, cols do
        local v = row[j]
        if v > vals[top_k] then
            local r = top_k
            while r > 1 and vals[r - 1] < v do
                vals[r], idx[r] = vals[r - 1], idx[r - 1]
                r = r - 1
            end
            vals[r], idx[r] = v, j
        end
    end
    n = top_k
else
    idx = {}
    for j = 1, cols do idx[j] = j end
    n = cols
end

local max_val = -huge
for r = 1, n do
    local v = row[idx[r]]
    if v > max_val then max_val = v end
end

local probs, sum = {}, 0
for r = 1, n do
    local e = exp((row[idx[r]] - max_val) / temperature)
    probs[r] = e
    sum = sum + e
end

local limit = n
if top_p > 0 and top_p < 1 then
    local acc = 0
    for r = 1, n do
        acc = acc + probs[r] / sum
        if acc >= top_p then limit = r break end
    end
end

local norm = 0
for r = 1, limit do norm = norm + probs[r] end
local target = math.random() * norm
local acc = 0
for r = 1, limit do
    acc = acc + probs[r]
    if acc >= target then return idx[r] end
end
return idx[limit]
end

function LMTrain:generate(seed_tokens, n, opts)
if self.model == nil then
    error("LMTrain:generate: no model yet -- call train:run() or train:load(path) first", 2)
end
opts = opts or {}
n = n or 1
if not is_int(n) or n < 0 then
    error("LMTrain:generate: n must be a non-negative integer, got " .. tostring(n), 2)
end

local was_string = type(seed_tokens) == "string"
if was_string then
    seed_tokens = self.tokenizer and self.tokenizer:encode(seed_tokens)
        or LMTrain.encode(seed_tokens)
end
if type(seed_tokens) ~= "table" or #seed_tokens == 0 then
    error("LMTrain:generate: need at least one seed token", 2)
end

local tokens = {}
for i = 1, #seed_tokens do
    local tok = seed_tokens[i]
    if not is_int(tok) or tok < 1 or tok > self.vocab then
        error(sformat("LMTrain:generate: seed token %d = %s is outside 1..%d",
            i, tostring(tok), self.vocab), 2)
    end
    tokens[i] = tok
end

local temperature = opts.temperature or self.temperature
local top_k = opts.top_k or self.top_k
local top_p = opts.top_p or self.top_p
local max_ctx = opts.ctx or opts.max_ctx or self.max_seq

local stop = opts.stop
if type(stop) == "string" then
    local enc = self.tokenizer and self.tokenizer:encode(stop) or LMTrain.encode(stop)
    if #enc ~= 1 then
        error(sformat("LMTrain:generate: stop string %q encodes to %d tokens; "
            .. "pass a single token id instead", stop, #enc), 2)
    end
    stop = enc[1]
end
if stop ~= nil and not is_int(stop) then
    error("LMTrain:generate: 'stop' must be a token id or a single-token string", 2)
end

local model = self.model
local prev_training = Tensor.training
Tensor.training = false

local use_cache = type(model.new_state) == "function" and type(model.step) == "function"
local ok, err = pcall(function()
    if use_cache then
        local state = model:new_state()
        local first = #tokens - max_ctx + 1
        if first < 1 then first = 1 end
        local logits
        for i = first, #tokens do
            logits = model:step(tokens[i], state)
        end
        for _ = 1, n do
            local next_id = pick_token(logits, self.vocab, temperature, top_k, top_p)
            tokens[#tokens + 1] = next_id
            if stop ~= nil and next_id == stop then break end
            if state.n >= max_ctx then
                state = model:new_state()
                local lo = #tokens - max_ctx + 1
                if lo < 1 then lo = 1 end
                for i = lo, #tokens do logits = model:step(tokens[i], state) end
            else
                logits = model:step(next_id, state)
            end
        end
    else
        Tensor.no_grad(function()
            for _ = 1, n do
                local window = tokens
                if #tokens > max_ctx then
                    window = {}
                    for i = #tokens - max_ctx + 1, #tokens do
                        window[#window + 1] = tokens[i]
                    end
                end
                local out = model:forward(window)
                local cols = out.data.cols
                local base = (out.data.rows - 1) * cols
                local row = {}
                for j = 1, cols do row[j] = out.data.data[base + j] end
                local next_id = pick_token(row, cols, temperature, top_k, top_p)
                tokens[#tokens + 1] = next_id
                if stop ~= nil and next_id == stop then break end
            end
        end)
    end
end)

Tensor.training = prev_training
if not ok then error(err, 2) end

if was_string then
    if self.tokenizer then return self.tokenizer:decode(tokens) end
    return LMTrain.decode(tokens)
end
return tokens
end

function LMTrain:parameters()
if self._dirty or self.model == nil then self:_build() end
return self.params
end

LMTrain.ARCH_KEYS = ARCH_KEYS
LMTrain.DEFAULTS = DEFAULTS
LMTrain._snapshot = snapshot
LMTrain._restore = restore

local ok_io, lmio = pcall(require, "lmio")
if ok_io and type(lmio) == "function" then lmio(LMTrain) end

setmetatable(LMTrain, { __call = function(_, ...) return LMTrain.new(...) end })

return LMTrain
