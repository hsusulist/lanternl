local Tensor = require("tensor2")
local matrix = require("matrix")
local Linear = require("linear")

local Attention = {}
Attention.__index = Attention

local DEFAULTS = {
    dim   = 64,
    heads = 1,
    scale = nil,
}

function Attention.new(config)
    config = config or {}
    local self = setmetatable({}, Attention)

    self.dim   = config.dim or DEFAULTS.dim
    self.heads = config.heads or DEFAULTS.heads
    assert(self.dim % self.heads == 0, "Attention: dim must be divisible by heads")

    self.head_dim = self.dim / self.heads
    self.scale = config.scale or (1 / math.sqrt(self.head_dim))

    self.q_proj = Linear.new{ in_dim = self.dim, out_dim = self.dim }
    self.k_proj = Linear.new{ in_dim = self.dim, out_dim = self.dim }
    self.v_proj = Linear.new{ in_dim = self.dim, out_dim = self.dim }
    self.o_proj = Linear.new{ in_dim = self.dim, out_dim = self.dim }

    return self
end

local function transpose_tensor(t)
    local out = Tensor.new(matrix.transpose(t.data), {t}, "transpose")
    out.backward_fn = function()
        local grad_t = matrix.transpose(out.grad)
        local total = t.grad.rows * t.grad.cols
        for i = 1, total do
            t.grad.data[i] = t.grad.data[i] + grad_t.data[i]
        end
    end
    return out
end

local function scale_tensor(t, s)
    local out = Tensor.new(matrix.scale(t.data, s), {t}, "scale")
    out.backward_fn = function()
        local total = t.grad.rows * t.grad.cols
        for i = 1, total do
            t.grad.data[i] = t.grad.data[i] + out.grad.data[i] * s
        end
    end
    return out
end

local function attend(q, k, v, scale)
    local k_t = transpose_tensor(k)
    local scores = Tensor.matmul(q, k_t)
    local scaled = scale_tensor(scores, scale)
    local weights = Tensor.softmax_rows(scaled)
    return Tensor.matmul(weights, v)
end

function Attention:forward(x)
    local q = self.q_proj:forward(x)
    local k = self.k_proj:forward(x)
    local v = self.v_proj:forward(x)

    if self.heads == 1 then
        local out = attend(q, k, v, self.scale)
        return self.o_proj:forward(out)
    end

    local head_outputs = {}
    for h = 1, self.heads do
        local start_col = (h - 1) * self.head_dim + 1
        local end_col = h * self.head_dim

        local qh = Tensor.slice_cols(q, start_col, end_col)
        local kh = Tensor.slice_cols(k, start_col, end_col)
        local vh = Tensor.slice_cols(v, start_col, end_col)

        head_outputs[h] = attend(qh, kh, vh, self.scale)
    end

    local merged = Tensor.concat_cols(head_outputs)
    return self.o_proj:forward(merged)
end

function Attention:parameters()
    local params = {}
    for _, p in ipairs(self.q_proj:parameters()) do table.insert(params, p) end
    for _, p in ipairs(self.k_proj:parameters()) do table.insert(params, p) end
    for _, p in ipairs(self.v_proj:parameters()) do table.insert(params, p) end
    for _, p in ipairs(self.o_proj:parameters()) do table.insert(params, p) end
    return params
end

return Attention
