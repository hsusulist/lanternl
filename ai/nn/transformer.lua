local Tensor = require("tensor2")
local matrix = require("matrix")
local Block = require("block")
local Linear = require("linear")

local Transformer = {}
Transformer.__index = Transformer

local DEFAULTS = {
    vocab  = 1000,
    dim    = 64,
    layers = 2,
    heads  = 4,
    ffn    = 128,
    eps    = 1e-5,
}

function Transformer.new(config)
    config = config or {}
    local self = setmetatable({}, Transformer)

    self.vocab  = config.vocab or DEFAULTS.vocab
    self.dim    = config.dim or DEFAULTS.dim
    self.layers = config.layers or DEFAULTS.layers
    self.heads  = config.heads or DEFAULTS.heads
    self.ffn    = config.ffn or DEFAULTS.ffn
    self.eps    = config.eps or DEFAULTS.eps

    local emb = matrix.new(self.vocab, self.dim)
    local total = self.vocab * self.dim
    for k = 1, total do
        emb.data[k] = (math.random() * 2 - 1) * 0.02
    end
    self.embed = Tensor.new(emb)

    self.blocks = {}
    for i = 1, self.layers do
        self.blocks[i] = Block.new{
            dim = self.dim, heads = self.heads, ffn = self.ffn, eps = self.eps
        }
    end

    self.lm_head = Linear.new{ in_dim = self.dim, out_dim = self.vocab }

    return self
end

function Transformer:embed_tokens(token_ids)
    local seq_len = #token_ids
    local out_data = matrix.new(seq_len, self.dim)

    for i, id in ipairs(token_ids) do
        local src_base = (id - 1) * self.dim
        local dst_base = (i - 1) * self.dim
        for j = 1, self.dim do
            out_data.data[dst_base + j] = self.embed.data.data[src_base + j]
        end
    end

    local embed_ref = self.embed
    local out = Tensor.new(out_data, {embed_ref}, "embed_lookup")
    out.backward_fn = function()
        for i, id in ipairs(token_ids) do
            local src_base = (id - 1) * self.dim
            local dst_base = (i - 1) * self.dim
            for j = 1, self.dim do
                embed_ref.grad.data[src_base + j] = embed_ref.grad.data[src_base + j] + out.grad.data[dst_base + j]
            end
        end
    end
    return out
end

function Transformer:forward(token_ids)
    local x = self:embed_tokens(token_ids)

    for i = 1, self.layers do
        x = self.blocks[i]:forward(x)
    end

    local logits = self.lm_head:forward(x)
    return logits
end

function Transformer:parameters()
    local params = { self.embed }
    for i = 1, self.layers do
        for _, p in ipairs(self.blocks[i]:parameters()) do
            table.insert(params, p)
        end
    end
    for _, p in ipairs(self.lm_head:parameters()) do
        table.insert(params, p)
    end
    return params
end

return Transformer