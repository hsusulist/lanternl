local Tensor = require("tensor2")
local matrix = require("matrix")
local ok_luatl, luatl_adapter = pcall(require, "luatl_adapter")
if not ok_luatl then luatl_adapter = { available = false } end

local RMSNorm = {}
RMSNorm.__index = RMSNorm

local DEFAULTS = {
    dim = 64,
    eps = 1e-5,
}

function RMSNorm.new(config)
    config = config or {}
    local self = setmetatable({}, RMSNorm)

    self.dim = config.dim or DEFAULTS.dim
    self.eps = config.eps or DEFAULTS.eps

    local w = matrix.new(1, self.dim)
    for j = 1, self.dim do
        w.data[j] = 1
    end
    self.weight = Tensor.new(w)

    return self
end

function RMSNorm:forward(x)
    local rows, cols = x.data.rows, x.data.cols
    local eps = self.eps
    local w = self.weight.data.data

    local out_data = matrix.new(rows, cols)
    local r_values = {}

    for i = 1, rows do
        local base = (i - 1) * cols
        local sum_sq = 0
        for j = 1, cols do
            local v = x.data.data[base + j]
            sum_sq = sum_sq + v * v
        end
        local mean_sq = sum_sq / cols
        r_values[i] = 1 / math.sqrt(mean_sq + eps)
    end

    if luatl_adapter.available then
        out_data.data = luatl_adapter.rmsnorm(x.data.data, rows, cols, w, eps)
    else
        for i = 1, rows do
            local base = (i - 1) * cols
            local r = r_values[i]
            for j = 1, cols do
                out_data.data[base + j] = x.data.data[base + j] * r * w[j]
            end
        end
    end

    local weight_ref = self.weight
    local out = Tensor.new(out_data, {x, weight_ref}, "rmsnorm")

    out.backward_fn = function()
        for i = 1, rows do
            local base = (i - 1) * cols
            local r = r_values[i]

            local S = 0
            for j = 1, cols do
                S = S + out.grad.data[base + j] * w[j] * x.data.data[base + j]
            end

            for j = 1, cols do
                local g = out.grad.data[base + j]
                local xij = x.data.data[base + j]

                local dx = r * w[j] * g - (r * r * r / cols) * xij * S
                x.grad.data[base + j] = x.grad.data[base + j] + dx

                weight_ref.grad.data[j] = weight_ref.grad.data[j] + g * xij * r
            end
        end
    end

    return out
end

function RMSNorm:parameters()
    return {self.weight}
end

return RMSNorm