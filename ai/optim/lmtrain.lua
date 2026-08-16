local Tensor = require("tensor2")
local Transformer = require("transformer")
local optim2 = require("optim2")
local lmlog = require("lmlog")
local ok_gpu_mod, lmtrain_gpu = pcall(require, "lmtrain_gpu")

local LMTrain = {}
LMTrain.__index = LMTrain

local IS_TTY = lmlog.is_tty
local wall = lmlog.now

-- Buffering policy now lives in lmlog.setup_stdout() (called on require of
-- lmlog).  It used to be `io.stdout:setvbuf("no")` duplicated twice here,
-- which left anyone driving lmlog directly at libc's mercy, and forced a
-- write(2) per byte on a pipe.  See PROBLEM 4 notes in lmlog.lua.

local floor, ceil = math.floor, math.ceil
local huge = math.huge
local exp, cos, pi = math.exp, math.cos, math.pi
local sformat = string.format

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
    backend = "cpu", log_every_sec = nil, graph = false,
    -- PROBLEM 1
    max_bytes = 8 * 1024 * 1024,
    -- PROBLEM 2
    hints = true,
    -- PROBLEM 3
    save_every = 1, keep_last = 1, save_best = true, save_at_end = true,
    resume = false, restore_best = "auto",
}

--- Keys that are legitimately absent (nil) but understood.  Used to be
--- declared and never read; now it backs the typo detector below.
local NILABLE = {
    data = true, log = true, stop_loss = true, stop_when = true,
    seed = true, on_stop = true, name = true, tokenizer = true,
    temperature = true, top_k = true, top_p = true,
    data_file = true, max_tokens = true,
    auto_save = true, save_format = true, precision = true,
    inplace_bar = true, stream = true, preset = true,
}

local ARCH_KEYS = {
    vocab = true, dim = true, layers = true, heads = true, ffn = true,
    causal = true, pos = true, rope_base = true, max_seq = true,
    tie = true, dropout = true, backend = true, graph = true,
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
    backend = "backend", device = "backend", engine = "backend",
    gpu = "backend", use_gpu = "backend",
    graph = "graph", cuda_graph = "graph",
    log_every_sec = "log_every_sec", log_interval = "log_every_sec",

    -- PROBLEM 1: data sources
    data_file = "data_file", datafile = "data_file", file = "data_file",
    text_file = "data_file", path = "data_file", corpus = "data_file",
    max_bytes = "max_bytes", byte_limit = "max_bytes", read_limit = "max_bytes",
    max_tokens = "max_tokens", token_limit = "max_tokens",

    -- PROBLEM 2
    hints = "hints", hint = "hints", tips = "hints",

    -- PROBLEM 3.  NOTE: 'save' is deliberately NOT aliased here.  In
    -- resolve_aliases the canonical name is `ALIASES[key] or key`, so an
    -- alias entry is consulted BEFORE the RESERVED check -- aliasing a
    -- reserved name would let a user silently shadow LMTrain:save.
    auto_save = "auto_save", autosave = "auto_save", auto_save_path = "auto_save",
    checkpoint = "auto_save", ckpt = "auto_save", save_to = "auto_save",
    save_every = "save_every", checkpoint_every = "save_every",
    save_interval = "save_every", ckpt_every = "save_every",
    keep_last = "keep_last", keep_checkpoints = "keep_last",
    max_checkpoints = "keep_last", rotate = "keep_last",
    save_best = "save_best", best_file = "save_best",
    save_at_end = "save_at_end", save_final = "save_at_end",
    resume = "resume", resume_from = "resume", auto_resume = "resume",
    restore_best = "restore_best",
    save_format = "save_format", format = "save_format",
    precision = "precision", prec = "precision",
}

local BAR_CHARS = {
    [1] = { fill = "#", empty = "-" },
    [2] = { fill = "\226\150\147", empty = "\226\150\145" },
    [3] = { fill = "=", empty = " " },
}

local POS_MODES = { rope = true, rotary = true, sinusoidal = true, sin = true,
                    absolute = true, abs = true, none = true, off = true }
local SCHEDULES = { cosine = true, linear = true, none = true, constant = true }

--- Every key the library understands, for the typo detector.
local KNOWN = {}
do
    for k in pairs(DEFAULTS) do KNOWN[k] = true end
    for k in pairs(NILABLE) do KNOWN[k] = true end
    for _, canon in pairs(ALIASES) do KNOWN[canon] = true end
end

local function is_int(x)
    return type(x) == "number" and x == floor(x) and x == x and x ~= huge and x ~= -huge
end

local function is_finite(x)
    return type(x) == "number" and x == x and x ~= huge and x ~= -huge
end

local function say(self, text)
    if self and self._reporter then return self._reporter:say(text) end
    return lmlog.say(text)
end

--- Cheap "did you mean" -- shared prefix + length proximity.  No full
--- edit distance; this only needs to catch save_evry / bath_size class
--- typos, and a wrong suggestion is worse than none.
local function suggest(key)
    local best, best_score = nil, 0
    for k in pairs(KNOWN) do
        local n = 0
        while n < #k and n < #key and k:sub(n + 1, n + 1) == key:sub(n + 1, n + 1) do
            n = n + 1
        end
        local score = n - math.abs(#k - #key) * 0.5
        if n >= 3 and score > best_score then best, best_score = k, score end
    end
    return best
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

-- ==========================================================================
--  PROBLEM 1 -- data sources
-- ==========================================================================

--- Is this an ai.Data instance?  Duck-typed against its REAL API (data.lua
--- exposes :batches(), :count(), :config() and a .files array) rather than
--- against a metatable identity, so a user's own loader with the same shape
--- works too, and so we do not hard-require ai/data/data.lua.
local function is_ai_data(v)
    return type(v) == "table"
       and type(v.batches) == "function"
       and type(v.count) == "function"
       and type(v.files) == "table"
end

--- Read a text file with a byte budget.
---
--- WHY CAPPED WHOLE-FILE AND NOT TRUE STREAMING:
--- build_windows() already materialises every training window as Lua tables
--- up front, and both run loops index `train_w` randomly for shuffling.  A
--- genuinely streaming trainer would need a different window generator and a
--- different shuffle strategy on both backends -- a much larger change, and
--- one that would diverge the two loops.  So we read the file in 1 MB blocks
--- (constant peak string memory, unlike f:read("*a") on a 2 GB corpus) and
--- stop at max_bytes.  The cap matters because the in-memory cost is roughly
--- 40x the text size: ~16 B per Lua array slot for the token list, then again
--- for each window's input and target copies.  8 MB of text is already
--- ~300 MB of RAM, which is the practical ceiling for "modest hardware".
--- Exceeding the cap warns loudly and truncates on a whitespace boundary --
--- it never silently eats a partial multi-byte character or BPE token.
local function read_text_file(path, max_bytes, verbose, owner)
    local f, err = io.open(path, "rb")
    if not f then
        error(sformat("LMTrain: cannot open data_file '%s': %s", tostring(path), tostring(err)), 3)
    end
    local size = f:seek("end") or 0
    f:seek("set", 0)

    local budget = max_bytes or huge
    local parts, got = {}, 0
    while got < budget do
        local want = math.min(1048576, budget - got)
        local blk = f:read(want)
        if not blk or #blk == 0 then break end
        parts[#parts + 1] = blk
        got = got + #blk
    end
    f:close()

    local text = table.concat(parts)
    local truncated = false
    if size > got then
        truncated = true
        -- back off to the last whitespace so we never split a UTF-8
        -- sequence or a BPE-mergeable word across the cut
        local cut = #text
        local floor_cut = math.max(1, #text - 4096)
        while cut > floor_cut and not string.find(string.sub(text, cut, cut), "%s") do
            cut = cut - 1
        end
        if cut > floor_cut then text = string.sub(text, 1, cut) end
    end

    if verbose then
        say(owner, sformat("[LMTrain] data_file %s: read %s of %s%s",
            path, lmlog.fmt_bytes(#text), lmlog.fmt_bytes(size),
            truncated and sformat(" (TRUNCATED at max_bytes; raise max_bytes to use more, "
                .. "e.g. max_bytes = %d)", math.min(size, (max_bytes or 0) * 4)) or ""))
    end
    if truncated and verbose then
        say(owner, "[LMTrain] note: only the first slice of the corpus is being trained on. "
            .. "This is a memory guard, not a bug -- every token loaded costs roughly 40x "
            .. "its size in Lua table overhead once windows are built.")
    end
    return text
end

--- Pull sequences out of an ai.Data instance using its documented API.
--- :batches() returns a list of batches, each a list of items; an item is a
--- raw line (string) when data.tokenizer is unset, or an already-encoded id
--- array when it is set.  We call it EXACTLY ONCE: it re-reads every file
--- from disk and re-shuffles on each call, so calling it twice would both
--- double the I/O and give inconsistent data.
local function sequences_from_ai_data(d, encode_text, verbose, owner, max_tokens)
    local batches = d:batches()
    if type(batches) ~= "table" then
        error("LMTrain: data:batches() did not return a table", 3)
    end
    local seqs, ntok = {}, 0
    for b = 1, #batches do
        local batch = batches[b]
        if type(batch) == "table" then
            for i = 1, #batch do
                local item = batch[i]
                local ids
                if type(item) == "string" then
                    ids = encode_text(item)
                elseif type(item) == "table" then
                    ids = item                  -- data.tokenizer already encoded it
                end
                if ids and #ids >= 2 then
                    seqs[#seqs + 1] = ids
                    ntok = ntok + #ids
                    if max_tokens and ntok >= max_tokens then
                        if verbose then
                            say(owner, sformat("[LMTrain] ai.Data: stopped at max_tokens = %d",
                                max_tokens))
                        end
                        return seqs, ntok
                    end
                end
            end
        end
    end
    if #seqs == 0 then
        error("LMTrain: the ai.Data instance produced no usable sequences "
            .. "(every line was empty or shorter than 2 tokens). Check its .files list.", 3)
    end
    if verbose then
        say(owner, sformat("[LMTrain] ai.Data: %d sequence(s), %s tokens, tokenizer=%s",
            #seqs, lmlog.fmt_count(ntok), d.tokenizer and "data's own" or "LMTrain's"))
    end
    return seqs, ntok
end

--- Normalise every accepted data form into token-id sequences.
--- Accepted:  string | {string,...} | {int,...} | {{int,...},...}
---            | ai.Data instance | data_file = "path" | {"a.txt","b.txt"}
local function coerce_data(config, owner)
    local out = {}
    for k, v in pairs(config) do out[k] = v end

    local verbose = (out.verbose ~= false)
    local tok = out.tokenizer

    local function encode_text(text)
        if tok then return tok:encode(text) end
        return LMTrain.encode(text)
    end
    local function default_vocab()
        if out.vocab == nil then
            out.vocab = tok and tok.vocab_size or 256
        end
    end

    -- ---- data_file --------------------------------------------------
    if out.data_file ~= nil then
        if out.data ~= nil then
            error("LMTrain: pass either 'data' or 'data_file', not both", 3)
        end
        local files = out.data_file
        if type(files) == "string" then files = { files } end
        if type(files) ~= "table" or #files == 0 then
            error("LMTrain: 'data_file' must be a path or a list of paths, got "
                .. type(out.data_file), 3)
        end
        local budget = out.max_bytes
        if budget == false then budget = nil end
        local seqs = {}
        for i = 1, #files do
            local per = budget and math.max(1, floor(budget / #files)) or nil
            local text = read_text_file(files[i], per, verbose, owner)
            if #text > 0 then
                local ids = encode_text(text)
                if #ids >= 2 then seqs[#seqs + 1] = ids end
            end
        end
        if #seqs == 0 then
            error("LMTrain: data_file produced no tokens (all files empty?)", 3)
        end
        out.data = (#seqs == 1) and seqs[1] or seqs
        out.data_file = nil          -- consumed; do not re-read on :config()
        out._source = "data_file"
        default_vocab()
        return out
    end

    -- ---- ai.Data instance -------------------------------------------
    if is_ai_data(out.data) then
        local d = out.data
        -- If the user gave LMTrain a tokenizer but ai.Data has none, share it
        -- so both sides agree on the id space.  Never override the Data's own.
        if tok and d.tokenizer == nil and type(d.config) == "function" then
            d:config{ tokenizer = tok }
        elseif d.tokenizer and tok == nil then
            out.tokenizer = d.tokenizer
            tok = d.tokenizer
        end
        out.data = sequences_from_ai_data(d, encode_text, verbose, owner, out.max_tokens)
        out._source = "ai.Data"
        default_vocab()
        return out
    end

    -- ---- existing paths, unchanged ----------------------------------
    if type(out.data) == "string" then
        out.data = encode_text(out.data)
        out._source = "string"
        default_vocab()
    elseif type(out.data) == "table" and type(out.data[1]) == "string" then
        local seqs = {}
        for i = 1, #out.data do seqs[i] = encode_text(out.data[i]) end
        out.data = seqs
        out._source = "strings"
        default_vocab()
    elseif type(out.data) == "table" then
        out._source = out._source or "ids"
    end
    return out
end

local function resolve_aliases(config, who)
    if type(config) ~= "table" then
        error(sformat("%s: expected a config table, got %s", who, type(config)), 3)
    end
    local resolved, origin = {}, {}
    local unknown
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
        if not KNOWN[canonical] and string.sub(key, 1, 1) ~= "_" then
            unknown = unknown or {}
            unknown[#unknown + 1] = key
        end
        resolved[canonical] = val
        origin[canonical] = key
    end

    -- Typo detection.  A misspelled 'save_evry' used to be silently accepted
    -- and the user lost the run anyway -- the exact outcome auto_save exists
    -- to prevent.  Warn (never error): unknown keys were legal before and
    -- some users stash their own metadata on the config.
    if unknown and config.hints ~= false and config.verbose ~= false then
        for i = 1, #unknown do
            local s = suggest(ALIASES[unknown[i]] or unknown[i])
            lmlog.say(sformat("[LMTrain] warning: unknown config key '%s'%s -- it will be "
                .. "stored but nothing reads it.", unknown[i],
                s and (", did you mean '" .. s .. "'?") or ""))
        end
    end

    if resolved.sched == true then resolved.sched = "cosine" end
    if resolved.sched == false then resolved.sched = "none" end
    if resolved.backend == true then resolved.backend = "gpu" end
    if resolved.backend == false then resolved.backend = "cpu" end
    if type(resolved.backend) == "string" then
        resolved.backend = string.lower(resolved.backend)
        if resolved.backend == "cuda" or resolved.backend == "luatl" then
            resolved.backend = "gpu"
        elseif resolved.backend == "lua" or resolved.backend == "tensor2" then
            resolved.backend = "cpu"
        end
    end
    return resolved
end

local function get_sequences(data, who)
    if type(data) ~= "table" then
        error(sformat("LMTrain%s: data must be a table of token ids, got %s", who, type(data)), 3)
    end
    if #data == 0 then
        -- was: error(sformat("...%s...", who, 3), 3) -- stray extra argument
        error(sformat("LMTrain%s: data is empty, expected at least 2 token ids", who), 3)
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
            error(sformat("LMTrain%s: sequence %d has %d token(s) but next-token training "
                .. "needs at least 2", who, s, len), 3)
        end
        for i = 1, len do
            local tok = seq[i]
            if not is_int(tok) then
                error(sformat("LMTrain%s: sequence %d, token %d must be an integer id, got %s",
                    who, s, i, tostring(tok)), 3)
            end
            if tok < 1 or tok > vocab then
                error(sformat("LMTrain%s: sequence %d, token %d = %d is outside 1..%d, "
                    .. "raise 'vocab' or remap your tokens", who, s, i, tok, vocab), 3)
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
            if n >= 1 then windows[#windows + 1] = { inp, tgt, n } end
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
            error(sformat("LMTrain%s: '%s' must be a positive integer, got %s",
                who, key, tostring(v)), 4)
        end
    end

    pos_int("vocab"); pos_int("dim"); pos_int("layers")
    pos_int("heads"); pos_int("ffn"); pos_int("epochs")
    pos_int("max_seq"); pos_int("batch_size")

    if self.max_seq < 2 then
        error(sformat("LMTrain%s: 'max_seq' must be at least 2 for next-token prediction, got %d",
            who, self.max_seq), 3)
    end
    if self.dim % self.heads ~= 0 then
        error(sformat("LMTrain%s: dim (%d) must be divisible by heads (%d)",
            who, self.dim, self.heads), 3)
    end
    if type(self.pos) ~= "string" or not POS_MODES[string.lower(self.pos)] then
        error(sformat("LMTrain%s: 'pos' must be 'rope', 'sinusoidal', 'absolute' or 'none', got %s",
            who, tostring(self.pos)), 3)
    end
    local pl = string.lower(self.pos)
    if (pl == "rope" or pl == "rotary") and floor(self.dim / self.heads) < 2 then
        error(sformat("LMTrain%s: RoPE needs dim/heads >= 2 (got dim=%d heads=%d)",
            who, self.dim, self.heads), 3)
    end
    if type(self.sched) ~= "string" or not SCHEDULES[string.lower(self.sched)] then
        error(sformat("LMTrain%s: 'sched' must be 'cosine', 'linear' or 'none', got %s",
            who, tostring(self.sched)), 3)
    end
    if type(self.causal) ~= "boolean" then
        error(sformat("LMTrain%s: 'causal' must be true or false, got %s",
            who, tostring(self.causal)), 3)
    end
    if not is_finite(self.lr) or self.lr <= 0 then
        error(sformat("LMTrain%s: 'lr' must be a positive finite number, got %s",
            who, tostring(self.lr)), 3)
    end
    if not is_finite(self.lr_min) or self.lr_min < 0 then
        error(sformat("LMTrain%s: 'lr_min' must be a non-negative number, got %s",
            who, tostring(self.lr_min)), 3)
    end
    if self.lr_min > self.lr then
        error(sformat("LMTrain%s: 'lr_min' (%g) cannot exceed 'lr' (%g)",
            who, self.lr_min, self.lr), 3)
    end
    if not is_finite(self.warmup) or self.warmup < 0 or self.warmup >= 1 then
        error(sformat("LMTrain%s: 'warmup' must be a fraction in [0,1), got %s",
            who, tostring(self.warmup)), 3)
    end
    if not is_finite(self.val_split) or self.val_split < 0 or self.val_split >= 1 then
        error(sformat("LMTrain%s: 'val_split' must be a fraction in [0,1), got %s",
            who, tostring(self.val_split)), 3)
    end
    if not is_finite(self.dropout) or self.dropout < 0 or self.dropout >= 1 then
        error(sformat("LMTrain%s: 'dropout' must be in [0,1), got %s",
            who, tostring(self.dropout)), 3)
    end
    if self.stride ~= nil and (not is_int(self.stride) or self.stride < 1) then
        error(sformat("LMTrain%s: 'stride' must be a positive integer, got %s",
            who, tostring(self.stride)), 3)
    end
    if not is_int(self.every) or self.every < 1 then
        error(sformat("LMTrain%s: 'every' must be a positive integer, got %s",
            who, tostring(self.every)), 3)
    end
    if not is_int(self.eval_every) or self.eval_every < 1 then
        error(sformat("LMTrain%s: 'eval_every' must be a positive integer, got %s",
            who, tostring(self.eval_every)), 3)
    end
    if not is_int(self.bar_len) or self.bar_len < 1 then
        error(sformat("LMTrain%s: 'bar_len' must be a positive integer, got %s",
            who, tostring(self.bar_len)), 3)
    end
    if not is_int(self.patience) or self.patience < 0 then
        error(sformat("LMTrain%s: 'patience' must be a non-negative integer, got %s",
            who, tostring(self.patience)), 3)
    end
    if not is_finite(self.min_delta) or self.min_delta < 0 then
        error(sformat("LMTrain%s: 'min_delta' must be a non-negative number, got %s",
            who, tostring(self.min_delta)), 3)
    end
    for _, key in ipairs({ "log", "stop_when", "on_stop" }) do
        if self[key] ~= nil and type(self[key]) ~= "function" then
            error(sformat("LMTrain%s: '%s' must be a function, got %s", who, key, type(self[key])), 3)
        end
    end
    if self.stop_loss ~= nil and not is_finite(self.stop_loss) then
        error(sformat("LMTrain%s: 'stop_loss' must be a finite number, got %s",
            who, tostring(self.stop_loss)), 3)
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
    local be = self.backend or "cpu"
    if be ~= "cpu" and be ~= "gpu" then
        error(sformat("LMTrain%s: 'backend' must be 'cpu' or 'gpu', got %s", who, tostring(be)), 3)
    end
    if self.log_every_sec ~= nil
       and (not is_finite(self.log_every_sec) or self.log_every_sec < 0) then
        error(sformat("LMTrain%s: 'log_every_sec' must be a non-negative number, got %s",
            who, tostring(self.log_every_sec)), 3)
    end

    -- ---- PROBLEM 3: auto-save config -------------------------------
    local as = self.auto_save
    if as ~= nil and as ~= false and as ~= true and type(as) ~= "string" then
        error(sformat("LMTrain%s: 'auto_save' must be a path, true or false, got %s",
            who, type(as)), 3)
    end
    if self.keep_last ~= nil and (not is_int(self.keep_last) or self.keep_last < 1) then
        error(sformat("LMTrain%s: 'keep_last' must be a positive integer, got %s",
            who, tostring(self.keep_last)), 3)
    end
    if self.save_every ~= nil and type(self.save_every) ~= "number"
       and type(self.save_every) ~= "string" then
        error(sformat("LMTrain%s: 'save_every' must be a number of epochs or a duration "
            .. "string like '10m', got %s", who, type(self.save_every)), 3)
    end
    if type(self.save_every) == "number"
       and (not is_int(self.save_every) or self.save_every < 1) then
        error(sformat("LMTrain%s: numeric 'save_every' must be a positive integer number of "
            .. "epochs, got %s", who, tostring(self.save_every)), 3)
    end
    local rb = self.restore_best
    if rb ~= nil and rb ~= true and rb ~= false and rb ~= "auto" then
        error(sformat("LMTrain%s: 'restore_best' must be true, false or 'auto', got %s",
            who, tostring(rb)), 3)
    end
    if self.max_bytes ~= nil and self.max_bytes ~= false
       and (not is_int(self.max_bytes) or self.max_bytes < 1) then
        error(sformat("LMTrain%s: 'max_bytes' must be a positive integer or false, got %s",
            who, tostring(self.max_bytes)), 3)
    end
end

local function snapshot(params)
    -- GPU shims carry no host storage (params[i].data.data is {}), so this
    -- would return a snapshot of nothing and _restore would then silently
    -- write nothing while reporting success.  Fail loudly instead.
    if params[1] and params[1].gpu then
        error("LMTrain._snapshot: this is a GPU model -- its parameters live in VRAM and "
            .. "params[i].data.data is an empty host mirror. Use gpu_model:snapshot() "
            .. "(GPU-to-GPU) or train:save(path) instead.", 2)
    end
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
    if params[1] and params[1].gpu then
        error("LMTrain._restore: GPU model -- use gpu_model:restore(snap)", 2)
    end
    for i = 1, #params do
        local d = params[i].data.data
        local t = snap[i]
        for k = 1, #t do d[k] = t[k] end
    end
    return true
end

-- ==========================================================================
--  PROBLEM 2 -- make the vocab/tokenizer story legible
-- ==========================================================================

--- How are ids actually being produced?  "tokenizer" | "byte" | "ids"
function LMTrain:_tokenization()
    if self.tokenizer then return "tokenizer" end
    local src = self._source
    if src == "string" or src == "strings" or src == "data_file"
       or src == "ai.Data" then
        return "byte"
    end
    return "ids"
end

--- Bytes of parameter memory burned by one embedding row.
local function row_cost(self)
    local per = (self.backend == "gpu") and 4 or 8
    -- weight + grad + adam m + adam v on the GPU; weight + grad on CPU
    local copies = (self.backend == "gpu") and 4 or 2
    if not self.tie then copies = copies * 2 end   -- untied head row too
    return self.dim * per * copies
end

function LMTrain:_vocab_report()
    if self.hints == false or self.verbose == false then return end
    if self._vocab_reported then return end
    self._vocab_reported = true

    local mode = self:_tokenization()
    local reach = self._max_token_id       -- highest id present in the data

    if mode == "tokenizer" then
        say(self, sformat("[LMTrain] tokenizer: BPE, vocab_size = %d (model vocab = %d)",
            self.tokenizer.vocab_size or -1, self.vocab))
        if self.tokenizer.vocab_size and self.vocab < self.tokenizer.vocab_size then
            say(self, sformat("[LMTrain] warning: vocab (%d) is SMALLER than the tokenizer's "
                .. "vocab_size (%d). Any id above %d will fail validation.",
                self.vocab, self.tokenizer.vocab_size, self.vocab))
        end
        return
    end

    if mode == "byte" then
        say(self, sformat("[LMTrain] tokenizer: none -- using byte-level ids (1..256). "
            .. "vocab = %d.", self.vocab))
        -- ---- the over-provisioning question, answered concretely -----
        if self.vocab > 256 then
            local dead = self.vocab - 256
            say(self, sformat(
                "[LMTrain] WARNING: vocab = %d but byte-level tokenization can only ever "
                .. "emit ids 1..256. The other %d embedding rows are DEAD: never looked up, "
                .. "never receive a gradient, and they cost ~%s of %s for nothing. They do "
                .. "NOT give you a bigger effective vocabulary.",
                self.vocab, dead, lmlog.fmt_bytes(dead * row_cost(self)),
                (self.backend == "gpu") and "VRAM" or "RAM"))
            say(self, "[LMTrain]   They are also actively harmful at sampling time: the tied "
                .. "head produces logits for those rows, so with temperature > 0 you can "
                .. "sample an id > 256 that LMTrain.decode cannot turn back into a byte. "
                .. "Sampling is therefore clamped to 1..256 for this model.")
            say(self, "[LMTrain]   To actually USE a bigger vocab, train a BPE tokenizer:")
            say(self, "[LMTrain]     local tok = require('tokenizer').new{ text_file = 'corpus.txt', "
                .. "vocab_size = 8000 }:train()")
            say(self, "[LMTrain]     LMTrain{ data_file = 'corpus.txt', tokenizer = tok, "
                .. "vocab = tok.vocab_size, ... }")
            self._sample_limit = 256
        elseif self.hints then
            say(self, "[LMTrain] hint: byte-level means ~1 token per character and a short "
                .. "effective context. A BPE tokenizer (require('tokenizer')) typically gives "
                .. "3-4x more text per window at the same max_seq. Pass "
                .. "tokenizer = tok, vocab = tok.vocab_size to use one.")
        end
        return
    end

    -- raw ids supplied by the user
    if reach and self.vocab > reach * 2 and self.vocab - reach > 64 then
        local dead = self.vocab - reach
        say(self, sformat("[LMTrain] note: vocab = %d but the highest id in your data is %d. "
            .. "%d embedding rows will never be trained (~%s wasted).%s",
            self.vocab, reach, dead, lmlog.fmt_bytes(dead * row_cost(self)),
            (self.backend == "gpu")
                and " On the GPU backend this is real VRAM and it also enlarges every "
                    .. "logits tensor ([B*T, vocab]), which is usually the single largest "
                    .. "activation in the model."
                or ""))
    end
end

function LMTrain:_build()
    if self.seed ~= nil then
        if not is_int(self.seed) then
            error("LMTrain: 'seed' must be an integer, got " .. tostring(self.seed), 2)
        end
        math.randomseed(self.seed)
    end
    check_config(self, "")

    -- ---- GPU-resident backend (opt-in, strictly additive) -----------
    if self.backend == "gpu" then
        if not ok_gpu_mod then
            error("LMTrain: backend = 'gpu' but lmtrain_gpu.lua failed to load: "
                  .. tostring(lmtrain_gpu), 2)
        end
        lmtrain_gpu.install(LMTrain)
        return self:_build_gpu()          -- fails loudly if luaTL is missing
    end
    self.gpu_model = nil

    local ok, model = pcall(Transformer.new, {
        vocab = self.vocab, dim = self.dim, layers = self.layers,
        heads = self.heads, ffn = self.ffn, causal = self.causal,
        pos = self.pos, rope_base = self.rope_base, max_seq = self.max_seq,
        tie = self.tie, dropout = self.dropout,
    })
    if not ok then
        error(sformat("LMTrain: failed to build Transformer{ vocab=%d, dim=%d, layers=%d, "
            .. "heads=%d, ffn=%d, pos=%s, causal=%s }: %s",
            self.vocab, self.dim, self.layers, self.heads, self.ffn,
            tostring(self.pos), tostring(self.causal), tostring(model)), 2)
    end
    if type(model) ~= "table" or type(model.forward) ~= "function"
       or type(model.parameters) ~= "function" then
        error("LMTrain: Transformer.new did not return a usable model "
            .. "(needs :forward and :parameters)", 2)
    end
    local ok_p, params = pcall(model.parameters, model)
    if not ok_p or type(params) ~= "table" or #params == 0 then
        error("LMTrain: model:parameters() failed or returned nothing to optimize: "
            .. tostring(params), 2)
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
    local self = setmetatable({}, LMTrain)
    config = coerce_data(config, self)

    self.history = {}
    self.best_loss = huge
    self.epoch = 0
    self.loss = nil
    self.val_loss = nil
    self.is_best = false
    self.stopped = nil
    self.elapsed = 0
    self.tokens_per_sec = 0
    self.inplace_bar = IS_TTY

    local preset = config.preset
    config.preset = nil

    for key, val in pairs(config) do self[key] = val end
    for key, default_val in pairs(DEFAULTS) do
        if self[key] == nil then self[key] = default_val end
    end

    self._auto_vocab = (config.vocab == nil)
    if self.data ~= nil then
        local max_id = scan_tokens(self.data)
        self._max_token_id = max_id
        if self._auto_vocab and max_id and max_id > self.vocab then
            self.vocab = max_id
        end
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
    self:_vocab_report()
    return self
end

function LMTrain:config(opts)
    opts = resolve_aliases(opts or {}, "LMTrain:config")
    if opts.tokenizer == nil and self.tokenizer ~= nil then opts.tokenizer = self.tokenizer end
    if opts.max_bytes == nil then opts.max_bytes = self.max_bytes end
    opts = coerce_data(opts, self)
    local rebuild = false
    for key, val in pairs(opts) do
        self[key] = val
        if ARCH_KEYS[key] or OPT_KEYS[key] then rebuild = true end
    end
    if opts.vocab ~= nil then self._auto_vocab = false end
    if opts.data ~= nil then
        local max_id = scan_tokens(self.data)
        self._max_token_id = max_id
        self._vocab_reported = false
        if self._auto_vocab and max_id and max_id > self.vocab then
            self.vocab = max_id
            rebuild = true
        end
    end
    if opts.lr ~= nil then
        if not is_finite(self.lr) or self.lr <= 0 then
            error("LMTrain:config: 'lr' must be a positive finite number, got "
                .. tostring(self.lr), 2)
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
    -- On the GPU path best_snapshot holds device handles.  Dropping the Lua
    -- reference alone strands that VRAM until a GC cycle happens to run the
    -- finalizers, with no way to reach it in the meantime; release it.
    if self.gpu_model and self.best_snapshot then
        pcall(function() self.gpu_model:release_snapshot(self.best_snapshot) end)
    end
    self.best_snapshot = nil
    self._dirty = true
    return self
end

function LMTrain:summary()
    return sformat(
        "LMTrain%s: vocab=%d dim=%d layers=%d heads=%d ffn=%d pos=%s causal=%s ctx=%d | "
        .. "%s lr=%g sched=%s bs=%d | epoch=%d loss=%s best=%s",
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
    if warm > 0 and step <= warm then return self.base_lr * (step / warm) end
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
    if self.gpu_model then return self:_evaluate_gpu(windows) end
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

--- Shared end-of-run best-weight restore policy, so both loops behave the
--- same.  `restore_best = "auto"` (default) preserves the historical
--- behaviour exactly; it just no longer does it silently.
function LMTrain:_should_restore_best()
    local mode = self.restore_best
    if mode == nil then mode = "auto" end
    if not self.keep_best or not self.best_snapshot then return false end
    if mode == true then return true end
    if mode == false then return false end
    if self.stopped ~= "epochs" then return true end
    -- Completed all epochs: historical behaviour keeps the FINAL weights even
    -- if an earlier epoch was better.  Surprising after overfitting, so say so
    -- rather than change results under existing users.
    local final = self.val_loss or self.loss
    if final and self.best_loss < huge and final > self.best_loss + 1e-12 then
        say(self, sformat("[LMTrain] note: final loss %.5f is worse than the best seen "
            .. "(%.5f) but the run completed all epochs, so the FINAL weights are kept "
            .. "(historical behaviour). Pass restore_best = true to keep the best instead.",
            final, self.best_loss))
    end
    return false
end

function LMTrain:run()
    if self.data == nil then
        error("LMTrain: no data set, use LMTrain{ data = {...} }, LMTrain{ data_file = 'f.txt' } "
            .. "or train:config{ data = {...} }", 2)
    end
    check_config(self, ":run")

    if self._dirty or self.model == nil or self.opt == nil then self:_build() end

    -- Resume BEFORE validating data: :load() overwrites self.vocab from the
    -- checkpoint header, so validating first would check the tokens against
    -- the wrong vocabulary.  :load() also calls _build() and clears _dirty.
    self:_maybe_resume()

    local sequences = validate_data(self.data, self.vocab, ":run")
    self:_vocab_report()

    if self.causal == false and self.verbose then
        say(self, "LMTrain: warning -- causal = false with next-token training lets every "
            .. "position attend to its own target. Loss will drop unrealistically fast.")
    end

    local all = build_windows(sequences, self.max_seq, self.stride)
    if #all == 0 then error("LMTrain: data produced no training windows", 2) end

    local train_w, val_w = all, {}
    if self.val_split > 0 and #all > 1 then
        local nval = floor(#all * self.val_split)
        if nval < 1 then nval = 1 end
        if nval >= #all then nval = #all - 1 end
        train_w, val_w = {}, {}
        for i = 1, #all - nval do train_w[i] = all[i] end
        for i = #all - nval + 1, #all do val_w[#val_w + 1] = all[i] end
    end

    -- GPU-resident path
    if self.gpu_model then return self:_run_gpu(train_w, val_w, build_windows) end

    local model, opt, params = self.model, self.opt, self.params
    local nwin = #train_w
    local bs = self.batch_size
    if bs > nwin then bs = nwin end
    local nbatch = ceil(nwin / bs)
    local total_steps = self.epochs * nbatch
    local order = {}
    for i = 1, nwin do order[i] = i end

    local started = wall()
    local reporter = lmlog.new{
        total_steps = total_steps,
        bar_style   = self.bar_style,
        bar_len     = self.bar_len,
        interval    = self.log_every_sec,
        inplace     = self.inplace_bar,
    }
    self._reporter = reporter
    local stalled = 0
    local gstep = 0
    self.stopped = nil

    self:_autosave_begin()

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

            if n_in_batch > 1 then optim2.scale_grads(params, 1 / n_in_batch) end

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
            reporter:tick(batch_tokens)
            if self.verbose then
                reporter:update({
                    step = (epoch - 1) * nbatch + b, epoch = epoch,
                    epochs = self.epochs, batch = b, nbatch = nbatch,
                    loss = batch_tokens > 0 and (batch_loss / batch_tokens) or 0,
                    lr = self.cur_lr, gnorm = last_norm,
                })
            end
            -- Time-based auto-save only; an epoch on a big corpus can be long.
            if self._as and self._as.seconds then self:_autosave_tick("batch") end
        end

        Tensor.training = false
        local avg_loss = sum_tok > 0 and (sum_loss / sum_tok) or huge
        self.loss = avg_loss
        self.accuracy = sum_tok > 0 and (100 * correct / sum_tok) or 0
        self.grad_norm = last_norm
        self.elapsed = wall() - started
        self.tokens_per_sec = reporter.rate

        -- Fresh-validation gate.  With eval_every > 1, self.val_loss on a
        -- non-eval epoch is STALE (from an earlier epoch), so comparing it to
        -- best_loss made is_best permanently false and inflated `patience`
        -- with phantom stalls.  Only judge on an epoch whose monitor is real.
        local did_eval = false
        if #val_w > 0 and (epoch % self.eval_every == 0 or epoch == self.epochs) then
            self.val_loss, self.val_accuracy = self:evaluate(val_w)
            did_eval = true
        end
        local use_val = (#val_w > 0)
        self._monitor_valid = (not use_val) or did_eval
        local monitor = use_val and self.val_loss or avg_loss

        self.history[#self.history + 1] = {
            epoch = epoch, loss = avg_loss, val_loss = self.val_loss,
            accuracy = self.accuracy, lr = self.cur_lr, grad_norm = last_norm,
            time = self.elapsed, skipped = skipped,
        }

        if not is_finite(avg_loss) then
            self.stopped = "diverged"
            self.is_best = false
            reporter:say(sformat("LMTrain: stopping at epoch %d, loss became %s. Weights were "
                .. "not updated with non-finite gradients; try a smaller lr.",
                epoch, tostring(avg_loss)))
            break
        end

        if self._monitor_valid then
            self.is_best = monitor < (self.best_loss - self.min_delta)
            if self.is_best then
                self.best_loss = monitor
                stalled = 0
                if self.keep_best then self.best_snapshot = snapshot(params) end
            else
                if monitor < self.best_loss then self.best_loss = monitor end
                stalled = stalled + 1
            end
        else
            self.is_best = false
        end

        if self.verbose and (epoch % self.every == 0 or epoch == self.epochs or epoch == 1) then
            if type(self.log) == "function" then
                local ok, err = pcall(self.log, self)
                if not ok then
                    error(sformat("LMTrain: custom log function failed at epoch %d: %s",
                        epoch, tostring(err)), 2)
                end
            else
                reporter:update({
                    step = epoch * nbatch, epoch = epoch, epochs = self.epochs,
                    batch = nbatch, nbatch = nbatch,
                    loss = avg_loss, acc = self.accuracy, lr = self.cur_lr,
                    gnorm = last_norm, val_loss = self.val_loss,
                    best = self.is_best,
                }, true)
            end
        end

        self:_autosave_tick("epoch")

        if self.stop_loss and self._monitor_valid and monitor <= self.stop_loss then
            self.stopped = "stop_loss"
            reporter:say(sformat("Stopped early at epoch %d: loss %.5f reached stop_loss %.5f",
                epoch, monitor, self.stop_loss))
            break
        end

        if self.stop_when then
            local ok, hit = pcall(self.stop_when, self)
            if not ok then
                error(sformat("LMTrain: stop_when failed at epoch %d: %s", epoch, tostring(hit)), 2)
            end
            if hit then
                self.stopped = "stop_when"
                reporter:say(sformat("Stopped early at epoch %d: stop_when condition met", epoch))
                break
            end
        end

        if self.patience > 0 and stalled >= self.patience then
            self.stopped = "patience"
            reporter:say(sformat("Stopped early at epoch %d: no improvement for %d epochs "
                .. "(best %.5f)", epoch, stalled, self.best_loss))
            break
        end
        collectgarbage("collect")
    end

    if self.stopped == nil then self.stopped = "epochs" end
    reporter:finish()

    -- Restore FIRST, then write the final checkpoint, so the file on disk
    -- contains the weights the user is actually left holding.
    if self:_should_restore_best() then
        if restore(self.params, self.best_snapshot) and self.verbose then
            say(self, sformat("Restored best weights (loss %.5f)", self.best_loss))
        end
    end
    self:_autosave_tick("final")
    self:_autosave_end()

    opt.lr = self.base_lr
    self.lr = self.base_lr
    Tensor.training = false
    self._reporter = nil

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

    -- PROBLEM 2: with byte-level ids and an over-provisioned vocab the head
    -- emits logits for rows that can never be decoded.  Sampling those raises
    -- deep inside LMTrain.decode; restrict the candidate set instead.  Only
    -- applied when tokenization is KNOWN byte-level -- clamping user-supplied
    -- raw ids would silently change their model's behaviour.
    local limit = self.vocab
    if self._sample_limit and self:_tokenization() == "byte" then
        limit = math.min(limit, self._sample_limit)
    end

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
            for i = first, #tokens do logits = model:step(tokens[i], state) end
            for _ = 1, n do
                local next_id = pick_token(logits, limit, temperature, top_k, top_p)
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
                    local ncand = math.min(cols, limit)
                    local row = {}
                    for j = 1, ncand do row[j] = out.data.data[base + j] end
                    local next_id = pick_token(row, ncand, temperature, top_k, top_p)
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

--- Release GPU resources deterministically.  Model:free() existed and was
--- never called from anywhere.
function LMTrain:close()
    if self.gpu_model then
        if self.best_snapshot then
            pcall(function() self.gpu_model:release_snapshot(self.best_snapshot) end)
            self.best_snapshot = nil
        end
        pcall(function() self.gpu_model:free() end)
        self.gpu_model, self.model, self.params = nil, nil, {}
        self._dirty = true
    end
    return self
end

LMTrain.ARCH_KEYS = ARCH_KEYS
LMTrain.DEFAULTS = DEFAULTS
LMTrain.ALIASES = ALIASES
LMTrain._snapshot = snapshot
LMTrain._restore = restore
LMTrain._build_windows = build_windows

local ok_io, lmio = pcall(require, "lmio")
if ok_io and type(lmio) == "function" then lmio(LMTrain) end
if ok_gpu_mod then lmtrain_gpu.install(LMTrain) end

setmetatable(LMTrain, { __call = function(_, ...) return LMTrain.new(...) end })

return LMTrain
