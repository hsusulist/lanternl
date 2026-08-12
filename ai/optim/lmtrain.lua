local Tensor = require("tensor2")
local Transformer = require("transformer")
local optim2 = require("optim2")

local LMTrain = {}
LMTrain.__index = LMTrain

local floor = math.floor
local huge = math.huge
local sformat = string.format
local srep = string.rep

local DEFAULTS = {
    vocab = 1000, dim = 64, layers = 2, heads = 4, ffn = 128,
    epochs = 200, lr = 0.05, lr_decay = true, lr_min = 0,
    bar_style = 1, bar_len = 20, every = 1,
    patience = 0, min_delta = 0,
    verbose = true,
}

local NILABLE = {
    data = true, log = true, stop_loss = true, stop_when = true,
    seed = true, on_stop = true, name = true,
}

local ARCH_KEYS = { vocab = true, dim = true, layers = true, heads = true, ffn = true }

local RESERVED = {
    new = true, run = true, config = true, bar = true,
    reset = true, summary = true, parameters = true,
}

local ALIASES = {
    e = "epochs", epoch = "epochs", epochs = "epochs", n = "epochs",
    iters = "epochs", steps = "epochs",
    lr = "lr", rate = "lr", learning_rate = "lr",
    lr_min = "lr_min", min_lr = "lr_min",
    decay = "lr_decay", lr_decay = "lr_decay",
    v = "vocab", vocab = "vocab", vocab_size = "vocab",
    d = "dim", dim = "dim", width = "dim", model_dim = "dim",
    l = "layers", layers = "layers", depth = "layers", nlayers = "layers",
    h = "heads", heads = "heads", nheads = "heads",
    f = "ffn", ffn = "ffn", ff = "ffn", hidden = "ffn",
    data = "data", tokens = "data", seq = "data", sequences = "data",
    bar = "bar_style", style = "bar_style", bar_style = "bar_style",
    bar_len = "bar_len", width_bar = "bar_len",
    log = "log", on_epoch = "log", callback = "log",
    every = "every", log_every = "every",
    stop = "stop_loss", stop_loss = "stop_loss", target_loss = "stop_loss",
    stop_when = "stop_when", ["until"] = "stop_when",
    patience = "patience", min_delta = "min_delta",
    verbose = "verbose", seed = "seed", name = "name", on_stop = "on_stop",
}

local BAR_CHARS = {
    [1] = { fill = "#", empty = "-" },
    [2] = { fill = "\226\150\147", empty = "\226\150\145" },
    [3] = { fill = "=", empty = " " },
}

local function is_int(x)
    return type(x) == "number" and x == floor(x) and x == x and x ~= huge and x ~= -huge
end

local function is_finite(x)
    return type(x) == "number" and x == x and x ~= huge and x ~= -huge
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
            error(sformat("%s: '%s' is a reserved method name and cannot be used as a config key",
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
    return resolved
end

local function make_pair(tokens)
    local input_ids, target_ids = {}, {}
    for i = 1, #tokens - 1 do
        input_ids[i] = tokens[i]
        target_ids[i] = tokens[i + 1]
    end
    return input_ids, target_ids
end

local function get_sequences(data, who)
    if type(data) ~= "table" then
        error(sformat("LMTrain%s: data must be a table of token ids, got %s", who, type(data)), 3)
    end
    if #data == 0 then
        error(sformat("LMTrain%s: data is empty, expected at least 2 token ids", who), 3)
    end
    if type(data[1]) == "table" then
        return data
    end
    return { data }
end

local function scan_tokens(data)
    local ok_seqs = (type(data) == "table") and data or nil
    if not ok_seqs then return nil end
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

local function loss_value(loss, epoch)
    if type(loss) ~= "table" then
        error(sformat("LMTrain: cross_entropy returned %s at epoch %d, expected a tensor",
            type(loss), epoch or 0), 3)
    end
    local d = loss.data
    if type(d) == "number" then return d end
    if type(d) == "table" then
        if type(d.data) == "table" and type(d.data[1]) == "number" then return d.data[1] end
        if type(d[1]) == "number" then return d[1] end
    end
    error(sformat("LMTrain: could not read a scalar loss value at epoch %d", epoch or 0), 3)
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
    if self.dim % self.heads ~= 0 then
        error(sformat("LMTrain%s: dim (%d) must be divisible by heads (%d)", who, self.dim, self.heads), 3)
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
    if not is_int(self.every) or self.every < 1 then
        error(sformat("LMTrain%s: 'every' must be a positive integer, got %s", who, tostring(self.every)), 3)
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
    if self.log ~= nil and type(self.log) ~= "function" then
        error(sformat("LMTrain%s: 'log' must be a function(trainer), got %s", who, type(self.log)), 3)
    end
    if self.stop_when ~= nil and type(self.stop_when) ~= "function" then
        error(sformat("LMTrain%s: 'stop_when' must be a function(trainer) -> boolean, got %s", who, type(self.stop_when)), 3)
    end
    if self.on_stop ~= nil and type(self.on_stop) ~= "function" then
        error(sformat("LMTrain%s: 'on_stop' must be a function(trainer), got %s", who, type(self.on_stop)), 3)
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
        error(sformat("LMTrain%s: unknown bar style %s (expected 1, 2, 3 or { fill = , empty = })",
            who, tostring(style)), 3)
    end
end

local function default_log(t)
    local style = t.bar_style
    if type(style) ~= "table" then style = BAR_CHARS[style] or BAR_CHARS[1] end
    local bar_len = t.bar_len or 20
    local total = t.epochs
    local filled = 0
    if total and total > 0 then
        filled = floor((t.epoch / total) * bar_len)
    end
    if filled < 0 then filled = 0 end
    if filled > bar_len then filled = bar_len end
    local bar_str = srep(style.fill, filled) .. srep(style.empty, bar_len - filled)
    local marker = t.is_best and " (best)" or ""
    print(sformat("[%s] Epoch %d/%d | Loss: %.5f | Acc: %.1f%%%s", bar_str, t.epoch, total, t.loss, t.accuracy or 0, marker))
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
        heads = self.heads, ffn = self.ffn,
    })
    if not ok then
        error(sformat("LMTrain: failed to build Transformer{ vocab=%d, dim=%d, layers=%d, heads=%d, ffn=%d }: %s",
            self.vocab, self.dim, self.layers, self.heads, self.ffn, tostring(model)), 2)
    end
    if type(model) ~= "table" or type(model.forward) ~= "function" or type(model.parameters) ~= "function" then
        error("LMTrain: Transformer.new did not return a usable model (needs :forward and :parameters)", 2)
    end
    local ok_p, params = pcall(model.parameters, model)
    if not ok_p or type(params) ~= "table" then
        error("LMTrain: model:parameters() failed: " .. tostring(params), 2)
    end
    if #params == 0 then
        error("LMTrain: model:parameters() returned no parameters, nothing to optimize", 2)
    end
    local ok_o, sgd = pcall(optim2.SGD, params, { lr = self.lr })
    if not ok_o or type(sgd) ~= "table" then
        error("LMTrain: optim2.SGD construction failed: " .. tostring(sgd), 2)
    end
    if type(sgd.zero_grad) ~= "function" or type(sgd.step) ~= "function" then
        error("LMTrain: optimizer must expose zero_grad() and step()", 2)
    end
    self.model = model
    self.params = params
    self.sgd = sgd
    self.base_lr = self.lr
    self._dirty = false
    return self
end

function LMTrain.new(config)
    config = resolve_aliases(config or {}, "LMTrain")
    if type(config.data) == "string" then
        local t = {}
        for i = 1, #config.data do t[i] = string.byte(config.data, i) end
        config.data = t
        if config.vocab == nil then config.vocab = 256 end
    end
    local self = setmetatable({}, LMTrain)

    self.history = {}
    self.best_loss = huge
    self.epoch = 0
    self.loss = nil
    self.is_best = false
    self.stopped = nil
    self.elapsed = 0

    for key, val in pairs(config) do
        self[key] = val
    end
    for key, default_val in pairs(DEFAULTS) do
        if self[key] == nil then self[key] = default_val end
    end
    for key in pairs(NILABLE) do
        if self[key] == nil and config[key] ~= nil then self[key] = config[key] end
    end

    self._auto_vocab = (config.vocab == nil)
    if self._auto_vocab and self.data ~= nil then
        local max_id = scan_tokens(self.data)
        if max_id then self.vocab = max_id end
    end

    self.base_lr = self.lr
    if config.preset == "auto" and self.data then
        local n = #self.data
        if type(self.data[1]) == "table" then n = #self.data[1] end
        if n < 100 then self.dim, self.layers, self.heads, self.ffn = 32, 1, 4, 64
        elseif n < 1000 then self.dim, self.layers, self.heads, self.ffn = 64, 2, 4, 128
        else self.dim, self.layers, self.heads, self.ffn = 128, 4, 8, 256 end
    end
    self:_build()
    return self
end

function LMTrain:config(opts)
    opts = resolve_aliases(opts or {}, "LMTrain:config")
    local rebuild = false
    for key, val in pairs(opts) do
        self[key] = val
        if ARCH_KEYS[key] then rebuild = true end
    end
    if opts.vocab ~= nil then
        self._auto_vocab = false
    end
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
        if self.sgd then self.sgd.lr = self.lr end
    end
    if rebuild then self._dirty = true end
    return self
end

function LMTrain:bar(opts)
    if type(opts) == "number" or type(opts) == "string" then
        opts = { type = opts }
    end
    opts = opts or {}
    if type(opts) ~= "table" then
        error("LMTrain:bar: expected a style id or an options table", 2)
    end
    local style = opts.type or opts.style or opts.bar
    if style ~= nil then
        if type(style) ~= "table" and BAR_CHARS[style] == nil then
            error(sformat("LMTrain:bar: unknown bar type %s (expected 1, 2, 3 or { fill = , empty = })",
                tostring(style)), 2)
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
    if opts.off == true then self.verbose = false end
    if opts.on == true then self.verbose = true end
    return self
end

function LMTrain:reset()
    self.history = {}
    self.best_loss = huge
    self.loss = nil
    self.epoch = 0
    self.is_best = false
    self.stopped = nil
    self.elapsed = 0
    self._dirty = true
    return self
end

function LMTrain:summary()
    return sformat(
        "LMTrain%s: vocab=%d dim=%d layers=%d heads=%d ffn=%d | epochs=%d lr=%g decay=%s | epoch=%d loss=%s best=%s",
        self.name and (" " .. tostring(self.name)) or "",
        self.vocab, self.dim, self.layers, self.heads, self.ffn,
        self.epochs, self.base_lr or self.lr, tostring(self.lr_decay and true or false),
        self.epoch,
        self.loss and sformat("%.5f", self.loss) or "n/a",
        self.best_loss < huge and sformat("%.5f", self.best_loss) or "n/a")
end

function LMTrain:run()
    if self.data == nil then
        error("LMTrain: no data set, use ai.LMTrain{ data = {...} } or train:config{ data = {...} }", 2)
    end
    check_config(self, ":run")
    local sequences = validate_data(self.data, self.vocab, ":run")

    if self._dirty or self.model == nil or self.sgd == nil then
        self:_build()
    end

    local cache = {}
    for i = 1, #sequences do
        local input_ids, target_ids = make_pair(sequences[i])
        cache[i] = { input_ids, target_ids }
    end
    local nseq = #cache
    local model, sgd = self.model, self.sgd
    local base_lr, lr_min, epochs = self.base_lr, self.lr_min, self.epochs
    local started = os.clock()
    local stalled = 0
    self.stopped = nil

    local total_correct = 0
    local total_tokens = 0

    for epoch = 1, epochs do
        self.epoch = epoch

        if self.lr_decay then
            local lr = base_lr * (1 - 0.9 * (epoch / epochs))
            if lr < lr_min then lr = lr_min end
            sgd.lr = lr
        else
            sgd.lr = base_lr
        end

        sgd.zero_grad()
        local total_loss = 0

        total_correct = 0
        total_tokens = 0

        for i = 1, nseq do
            local pair = cache[i]
            local logits = model:forward(pair[1])
            local loss = Tensor.cross_entropy(logits, pair[2])
            loss:backward()
            local correct = 0
            for r = 1, logits.data.rows do
                local base = (r - 1) * logits.data.cols
                local best_id, best_val = 1, -math.huge
                for c = 1, logits.data.cols do
                    local v = logits.data.data[base + c]
                    if v > best_val then best_val, best_id = v, c end
                end
                if best_id == pair[2][r] then correct = correct + 1 end
            end
            total_correct = total_correct + correct
            total_tokens = total_tokens + logits.data.rows
            total_loss = total_loss + loss_value(loss, epoch)
        end

        sgd.step()

        local avg_loss = total_loss / nseq
        self.loss = avg_loss
        self.accuracy = 100 * total_correct / total_tokens
        self.history[#self.history + 1] = avg_loss
        self.elapsed = os.clock() - started

        if not is_finite(avg_loss) then
            self.stopped = "diverged"
            self.is_best = false
            if self.verbose then
                print(sformat("LMTrain: stopping at epoch %d, loss became %s (lr = %g). Try a smaller lr or lr_decay = true.",
                    epoch, tostring(avg_loss), sgd.lr))
            end
            break
        end

        self.is_best = avg_loss < (self.best_loss - self.min_delta)
        if self.is_best then
            self.best_loss = avg_loss
            stalled = 0
        else
            if avg_loss < self.best_loss then self.best_loss = avg_loss end
            stalled = stalled + 1
        end

        if self.verbose and (epoch % self.every == 0 or epoch == epochs or epoch == 1) then
            if type(self.log) == "function" then
                local ok, err = pcall(self.log, self)
                if not ok then
                    error(sformat("LMTrain: custom log function failed at epoch %d: %s", epoch, tostring(err)), 2)
                end
            else
                default_log(self)
            end
        end

        if self.stop_loss and avg_loss <= self.stop_loss then
            self.stopped = "stop_loss"
            if self.verbose then
                print(sformat("Stopped early at epoch %d: loss %.5f reached stop_loss %.5f",
                    epoch, avg_loss, self.stop_loss))
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
                    print(sformat("Stopped early at epoch %d: stop_when condition met", epoch))
                end
                break
            end
        end

        if self.patience > 0 and stalled >= self.patience then
            self.stopped = "patience"
            if self.verbose then
                print(sformat("Stopped early at epoch %d: no improvement for %d epochs (best %.5f)",
                    epoch, stalled, self.best_loss))
            end
            break
        end
    end

    if self.stopped == nil then self.stopped = "epochs" end
    sgd.lr = base_lr
    self.lr = base_lr

    if type(self.on_stop) == "function" then
        pcall(self.on_stop, self)
    end

    return self.model
end

function LMTrain:generate(seed_tokens, n)
    local was_string = type(seed_tokens) == "string"
    if was_string then
        local t = {}
        for i = 1, #seed_tokens do t[i] = string.byte(seed_tokens, i) end
        seed_tokens = t
    end
    local tokens = {}
    for i = 1, #seed_tokens do tokens[i] = seed_tokens[i] end

    for _ = 1, n do
        local logits = self.model:forward(tokens)
        local last_row = (logits.data.rows - 1) * logits.data.cols
        local best_id, best_val = 1, -math.huge
        for j = 1, logits.data.cols do
            local v = logits.data.data[last_row + j]
            if v > best_val then best_val, best_id = v, j end
        end
        tokens[#tokens + 1] = best_id
    end
    
    if was_string then
        local chars = {}
        for i = 1, #tokens do
            chars[i] = string.char(tokens[i])
        end
        return table.concat(chars)
    end
    
    return tokens
end

function LMTrain:save(path)
    local f = io.open(path, "w")
    f:write(string.format("vocab=%d dim=%d layers=%d heads=%d ffn=%d\n",
        self.vocab, self.dim, self.layers, self.heads, self.ffn))
    for i = 1, #self.params do
        local d = self.params[i].data.data
        local total = self.params[i].data.rows * self.params[i].data.cols
        local parts = {}
        for k = 1, total do parts[k] = tostring(d[k]) end
        f:write(table.concat(parts, ",") .. "\n")
    end
    f:close()
    print("Saved to " .. path)
    return self
end

function LMTrain:load(path)
    local f = io.open(path, "r")
    if not f then error("LMTrain:load: cannot open " .. path, 2) end
    local header = f:read("*l")
    local vocab, dim, layers, heads, ffn =
        header:match("vocab=(%d+) dim=(%d+) layers=(%d+) heads=(%d+) ffn=(%d+)")
    self.vocab, self.dim, self.layers, self.heads, self.ffn =
        tonumber(vocab), tonumber(dim), tonumber(layers), tonumber(heads), tonumber(ffn)
    self:_build()
    for i = 1, #self.params do
        local line = f:read("*l")
        local d, j = self.params[i].data.data, 0
        for num_str in line:gmatch("[^,]+") do
            j = j + 1
            d[j] = tonumber(num_str)
        end
    end
    f:close()
    print("Loaded from " .. path)
    return self
end

function LMTrain:push(repo_id, token)
    self:save("lanternl_model.txt")
    local cmd = token
        and string.format('huggingface-cli upload %s lanternl_model.txt lanternl_model.txt --token %s', repo_id, token)
        or string.format('huggingface-cli upload %s lanternl_model.txt lanternl_model.txt', repo_id)
    local ok = os.execute(cmd)
    if not ok then
        print("LMTrain:push: upload failed — check huggingface-cli is installed and you're logged in")
    end
    return ok
end

function LMTrain:parameters()
    if self._dirty or self.model == nil then self:_build() end
    return self.params
end

setmetatable(LMTrain, {
    __call = function(_, ...) return LMTrain.new(...) end
})

return LMTrain
