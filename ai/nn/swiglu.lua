local Tensor = require("tensor2")
local Linear = require("linear")

local SwiGLU = {}
SwiGLU.__index = SwiGLU

local DEFAULTS = {
    dim = 64,
    ffn = 128,
}

function SwiGLU.new(config)
    config = config or {}
    local self = setmetatable({}, SwiGLU)

    self.dim = config.dim or DEFAULTS.dim
    self.ffn = config.ffn or DEFAULTS.ffn

    self.gate = Linear.new{ in_dim = self.dim, out_dim = self.ffn }
    self.up   = Linear.new{ in_dim = self.dim, out_dim = self.ffn }
    self.down = Linear.new{ in_dim = self.ffn, out_dim = self.dim }

    return self
end

function SwiGLU:forward(x)
    local g = self.gate:forward(x)
    local u = self.up:forward(x)
    local activated = Tensor.silu(g)
    local combined = Tensor.mul(activated, u)
    return self.down:forward(combined)
end

function SwiGLU:parameters()
    local params = {}
    for _, p in ipairs(self.gate:parameters()) do table.insert(params, p) end
    for _, p in ipairs(self.up:parameters()) do table.insert(params, p) end
    for _, p in ipairs(self.down:parameters()) do table.insert(params, p) end
    return params
end

return SwiGLU