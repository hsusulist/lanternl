local Tensor = require("tensor2")
local Transformer = require("transformer")
local optim2 = require("optim2")

local LMTrain = {}
LMTrain.__index = LMTrain

local DEFAULTS = {
    vocab = 1000, dim = 64, layers = 2, heads = 4, ffn = 128,
    epochs = 200, lr = 0.05, lr_decay = true,
    bar = 1, log = nil,
    stop_loss = nil, stop_when = nil,
    verbose = true,
}

-- short-name aliases -> canonical config key
local ALIASES = {
    e = "epochs", epoch = "epochs", epochs = "epochs",
    lr = "lr", rate = "lr",
    v = "vocab", vocab = "vocab",
    d = "dim", dim = "dim",
    l = "layers", layers = "layers",
    h = "heads", heads = "heads",
    f = "ffn", ffn = "ffn",
    data = "data", tokens = "data",
}

local function resolve_aliases(config)
    local resolved = {}
    for key, val in pairs(config) do
        local canonical = ALIASES[key] or key
        resolved[canonical] = val
    end
    return resolved
end

-- Turn a flat token list into {input_ids, target_ids} (next-token prediction)
local function make_pair(tokens)
    local input_ids, target_ids = {}, {}
    for i = 1, #tokens - 1 do
        input_ids[i] = tokens[i]
        target_ids[i] = tokens[i + 1]
    end
    return input_ids, target_ids
end

function LMTrain.new(config)
    config = resolve_aliases(config or {})
    local self = setmetatable({}, LMTrain)

    for key, default_val in pairs(DEFAULTS) do
        self[key] = config[key]
        if self[key] == nil then self[key] = default_val end
    end

    self.base_lr = self.lr
    self.model = Transformer.new{
        vocab = self.vocab, dim = self.dim, layers = self.layers,
        heads = self.heads, ffn = self.ffn,
    }
    self.sgd = optim2.SGD(self.model:parameters(), { lr = self.lr })

    self.loss = nil
    self.epoch = 0
    self.history = {}
    self.best_loss = math.huge

    return self
end

function LMTrain:config(opts)
    opts = resolve_aliases(opts or {})
    for key, val in pairs(opts) do
        self[key] = val
    end
    if opts.lr then self.base_lr = opts.lr end
    return self
end

function LMTrain:bar(opts)
    opts = opts or {}
    if opts.type then self.bar_style = opts.type end
    return self
end

-- ===== Built-in log renderers =====
local BAR_CHARS = {
    [1] = { fill = "#", empty = "-" },
    [2] = { fill = "▓", empty = "░" },
    [3] = { fill = "=", empty = " " },
}

local function default_log(t)
    local style = BAR_CHARS[t.bar] or BAR_CHARS[1]
    local bar_len = 20
    local filled = math.floor((t.epoch / t.epochs) * bar_len)
    local bar_str = string.rep(style.fill, filled) .. string.rep(style.empty, bar_len - filled)
    local marker = t.is_best and " (best)" or ""
    print(string.format("[%s] Epoch %d/%d | Loss: %.5f%s", bar_str, t.epoch, t.epochs, t.loss, marker))
end

local function get_sequences(data)
    if type(data[1]) == "table" then
        return data -- already a list of sequences
    end
    return { data } -- single flat sequence -> wrap as one sequence
end

function LMTrain:run()
    assert(self.data, "LMTrain: no data set, use LMTrain{ data = {...} } or train:config{ data = {...} }")
    local sequences = get_sequences(self.data)

    for epoch = 1, self.epochs do
        self.epoch = epoch

        if self.lr_decay then
            self.sgd.lr = self.base_lr * (1 - 0.9 * (epoch / self.epochs))
        end

        self.sgd.zero_grad()
        local total_loss = 0

        for _, seq in ipairs(sequences) do
            local input_ids, target_ids = make_pair(seq)
            local logits = self.model:forward(input_ids)
            local loss = Tensor.cross_entropy(logits, target_ids)
            loss:backward()
            total_loss = total_loss + loss.data.data[1]
        end

        self.sgd.step()

        local avg_loss = total_loss / #sequences
        self.loss = avg_loss
        table.insert(self.history, avg_loss)

        self.is_best = avg_loss < self.best_loss
        if self.is_best then self.best_loss = avg_loss end

        if self.verbose then
            if type(self.log) == "function" then
                self.log(self)
            else
                default_log(self)
            end
        end

        if self.stop_loss and avg_loss <= self.stop_loss then
            if self.verbose then print("Stopped early: loss reached stop_loss") end
            break
        end
        if self.stop_when and self.stop_when(self) then
            if self.verbose then print("Stopped early: stop_when condition met") end
            break
        end
    end

    return self.model
end

setmetatable(LMTrain, {
    __call = function(_, ...) return LMTrain.new(...) end
})

return LMTrain