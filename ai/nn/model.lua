local model = {}
model.__index = model

local DEFAULTS = {
    vocab    = 1000,
    dim      = 64,
    layers   = 2,
    heads    = 4,
    kv_heads = 2,
    head_dim = 16,
    ffn      = 128,
    seq_len  = 32,
}

function model.new(config)
    local self = setmetatable({}, model)
    config = config or {}

    for key, default_val in pairs(DEFAULTS) do
        self[key] = config[key] or default_val
    end

    return self
end

function model:print_config()
    print("Model config:")
    print("  vocab    =", self.vocab)
    print("  dim      =", self.dim)
    print("  layers   =", self.layers)
    print("  heads    =", self.heads)
    print("  kv_heads =", self.kv_heads)
    print("  head_dim =", self.head_dim)
    print("  ffn      =", self.ffn)
    print("  seq_len  =", self.seq_len)
end

return model