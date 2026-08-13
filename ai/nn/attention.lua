local Tensor = require("tensor2")
local matrix = require("matrix")
local Linear = require("linear")
local RoPE = require("rope")

local Attention = {}
Attention.__index = Attention

local floor = math.floor

local DEFAULTS = {
    dim       = 64,
    heads     = 1,
    scale     = nil,
    causal    = true,     -- decoder-style masking on by default
    pos       = "rope",   -- "rope" | "none"
    rope_base = 10000,
}

function Attention.new(config)
    config = config or {}
    local self = setmetatable({}, Attention)

    self.dim   = config.dim or DEFAULTS.dim
    self.heads = config.heads or DEFAULTS.heads
    assert(self.dim % self.heads == 0, "Attention: dim must be divisible by heads")

    -- floor() keeps this an integer under Lua 5.3+ float division
    self.head_dim = floor(self.dim / self.heads)
    self.scale = config.scale or (1 / math.sqrt(self.head_dim))

    -- causal defaults to TRUE: without it, forwarding a whole sequence at
    -- once lets position i read position i+1, which is literally the label
    -- in next-token training.
    self.causal = (config.causal ~= false)

    self.pos = config.pos or DEFAULTS.pos
    if self.pos == true then self.pos = "rope" end
    if self.pos == false then self.pos = "none" end
    if self.pos ~= "rope" and self.pos ~= "none" then
        error("Attention: pos must be 'rope' or 'none' (sinusoidal is added in transformer.lua), got "
            .. tostring(self.pos), 2)
    end

    if self.pos == "rope" then
        if self.head_dim < 2 then
            error(string.format("Attention: RoPE needs head_dim >= 2, got dim=%d / heads=%d",
                self.dim, self.heads), 2)
        end
        self.rope = RoPE.new{
            head_dim = self.head_dim,
            base     = config.rope_base or DEFAULTS.rope_base,
            max_seq  = config.max_seq,
        }
    end

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

-- q is expected PRE-SCALED by 1/sqrt(head_dim); scaling q (n x d) once is
-- cheaper than scaling the (n x n) score matrix per head, and identical.
local function attend(q, k, v, mask_opts)
    local k_t = transpose_tensor(k)
    local scores = Tensor.matmul(q, k_t)
    local weights = Tensor.softmax_rows(scores, mask_opts)
    return Tensor.matmul(weights, v)
end

-- opts.offset: absolute position of row 1 (for a future KV cache). Default 0.
function Attention:forward(x, opts)
    local offset = (opts and opts.offset) or 0

    local mask_opts = nil
    if self.causal then
        if offset == 0 then
            mask_opts = true
        else
            mask_opts = { causal = true, offset = offset }
        end
    end

    -- fold the 1/sqrt(head_dim) into q before the head split;
    -- a scalar commutes with the RoPE rotation, so order is irrelevant.
    local q = scale_tensor(self.q_proj:forward(x), self.scale)
    local k = self.k_proj:forward(x)
    local v = self.v_proj:forward(x)

    local cos_rows, sin_rows, npairs
    if self.rope then
        cos_rows, sin_rows = self.rope:rows(x.data.rows, offset)
        npairs = self.rope.pairs
    end

    if self.heads == 1 then
        if self.rope then
            q = Tensor.rope(q, cos_rows, sin_rows, npairs)
            k = Tensor.rope(k, cos_rows, sin_rows, npairs)
        end
        local out = attend(q, k, v, mask_opts)
        return self.o_proj:forward(out)
    end

    local head_outputs = {}
    for h = 1, self.heads do
        local start_col = (h - 1) * self.head_dim + 1
        local end_col = h * self.head_dim

        local qh = Tensor.slice_cols(q, start_col, end_col)
        local kh = Tensor.slice_cols(k, start_col, end_col)
        local vh = Tensor.slice_cols(v, start_col, end_col)

        if self.rope then
            qh = Tensor.rope(qh, cos_rows, sin_rows, npairs)
            kh = Tensor.rope(kh, cos_rows, sin_rows, npairs)
            -- note: never applied to V
        end

        head_outputs[h] = attend(qh, kh, vh, mask_opts)
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
