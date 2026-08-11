local Tensor = require("tensor2")

local Linear = {}
Linear.__index = Linear

function Linear.new(config)
    local self = setmetatable({}, Linear)

    self.in_dim = config.in_dim
    self.out_dim = config.out_dim
    assert(self.in_dim, "Linear: need config.in_dim")
    assert(self.out_dim, "Linear: need config.out_dim")

    self.weight = Tensor.random(self.in_dim, self.out_dim, 0.1)

    return self
end

function Linear:forward(x)
    return Tensor.matmul(x, self.weight)
end

function Linear:parameters()
    return {self.weight}
end

return Linear