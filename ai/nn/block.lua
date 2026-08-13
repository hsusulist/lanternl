local Tensor = require("tensor2")
local Attention = require("attention")
local RMSNorm = require("rmsnorm")
local SwiGLU = require("swiglu")

local Block = {}
Block.__index = Block

local DEFAULTS = {
    dim       = 64,
    heads     = 4,
    ffn       = 128,
    eps       = 1e-5,
    causal    = true,
    pos       = "rope",
    rope_base = 10000,
}

function Block.new(config)
    config = config or {}
    local self = setmetatable({}, Block)

    self.dim   = config.dim or DEFAULTS.dim
    self.heads = config.heads or DEFAULTS.heads
    self.ffn   = config.ffn or DEFAULTS.ffn
    self.eps   = config.eps or DEFAULTS.eps
    self.causal = (config.causal ~= false)
    self.pos   = config.pos or DEFAULTS.pos

    self.norm1 = RMSNorm.new{ dim = self.dim, eps = self.eps }
    self.attn  = Attention.new{
        dim       = self.dim,
        heads     = self.heads,
        causal    = self.causal,
        pos       = self.pos,
        rope_base = config.rope_base or DEFAULTS.rope_base,
        max_seq   = config.max_seq,
    }
    self.norm2 = RMSNorm.new{ dim = self.dim, eps = self.eps }
    self.ffn_layer = SwiGLU.new{ dim = self.dim, ffn = self.ffn }

    return self
end

function Block:forward(x, opts)
    local attn_out = self.attn:forward(self.norm1:forward(x), opts)
    local x1 = Tensor.add(x, attn_out)

    local ffn_out = self.ffn_layer:forward(self.norm2:forward(x1))
    local x2 = Tensor.add(x1, ffn_out)

    return x2
end

function Block:parameters()
    local params = {}
    for _, p in ipairs(self.norm1:parameters()) do table.insert(params, p) end
    for _, p in ipairs(self.attn:parameters()) do table.insert(params, p) end
    for _, p in ipairs(self.norm2:parameters()) do table.insert(params, p) end
    for _, p in ipairs(self.ffn_layer:parameters()) do table.insert(params, p) end
    return params
end

return Block
