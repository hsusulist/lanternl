local Tensor = require("tensor2")
local matrix = require("matrix")
local Block = require("block")
local Linear = require("linear")

local Transformer = {}
Transformer.__index = Transformer

local floor = math.floor

local DEFAULTS = {
    vocab     = 1000,
    dim       = 64,
    layers    = 2,
    heads     = 4,
    ffn       = 128,
    eps       = 1e-5,
    causal    = true,
    pos       = "rope",
    rope_base = 10000,
}

-- "rope" (default), "sinusoidal" (legacy additive PE), or "none".
local function normalize_pos(p)
    if p == nil then return DEFAULTS.pos end
    if p == true then return "rope" end
    if p == false then return "none" end
    p = string.lower(tostring(p))
    if p == "rope" or p == "rotary" then return "rope" end
    if p == "sinusoidal" or p == "sin" or p == "absolute" or p == "abs" then return "sinusoidal" end
    if p == "none" or p == "off" then return "none" end
    error("Transformer: unknown pos mode '" .. p .. "' (expected 'rope', 'sinusoidal' or 'none')", 3)
end

function Transformer.new(config)
    config = config or {}
    local self = setmetatable({}, Transformer)

    self.vocab  = config.vocab or DEFAULTS.vocab
    self.dim    = config.dim or DEFAULTS.dim
    self.layers = config.layers or DEFAULTS.layers
    self.heads  = config.heads or DEFAULTS.heads
    self.ffn    = config.ffn or DEFAULTS.ffn
    self.eps    = config.eps or DEFAULTS.eps
    self.causal = (config.causal ~= false)
    self.pos    = normalize_pos(config.pos)
    self.rope_base = config.rope_base or DEFAULTS.rope_base
    self.max_seq   = config.max_seq

    local emb = matrix.new(self.vocab, self.dim)
    local total = self.vocab * self.dim
    for k = 1, total do
        emb.data[k] = (math.random() * 2 - 1) * 0.02
    end
    self.embed = Tensor.new(emb)

    -- Additive sinusoidal PE has unit amplitude while embeddings start at
    -- +/-0.02, so the position signal would swamp the token signal. The
    -- classic fix is to scale embeddings by sqrt(dim) (Vaswani et al.);
    -- pos_scale lets you damp the PE further. RoPE needs neither, which is
    -- one of the reasons it is the default here.
    self.embed_scale = 1
    self.pos_scale = config.pos_scale or 1
    if self.pos == "sinusoidal" then
        local positional = require("positional")
        self.positional = positional.new{ dim = self.dim, seq_len = self.max_seq or 0 }
        self.embed_scale = math.sqrt(self.dim)
    end

    self.blocks = {}
    for i = 1, self.layers do
        self.blocks[i] = Block.new{
            dim       = self.dim,
            heads     = self.heads,
            ffn       = self.ffn,
            eps       = self.eps,
            causal    = self.causal,
            pos       = (self.pos == "rope") and "rope" or "none",
            rope_base = self.rope_base,
            max_seq   = self.max_seq,
        }
    end

    self.lm_head = Linear.new{ in_dim = self.dim, out_dim = self.vocab }

    return self
end

function Transformer:embed_tokens(token_ids, offset)
    offset = offset or 0
    local seq_len = #token_ids
    if seq_len == 0 then
        error("Transformer: embed_tokens got an empty token list", 2)
    end

    local dim = self.dim
    local out_data = matrix.new(seq_len, dim)
    local src = self.embed.data.data
    local dst = out_data.data
    local escale = self.embed_scale
    local pscale = self.pos_scale

    local pe = nil
    if self.pos == "sinusoidal" then
        pe = self.positional:rows(seq_len + offset)
    end

    for i = 1, seq_len do
        local id = token_ids[i]
        if type(id) ~= "number" or id ~= floor(id) or id < 1 or id > self.vocab then
            error(string.format(
                "Transformer: token %d is %s, expected an integer id in 1..%d",
                i, tostring(id), self.vocab), 2)
        end
        local src_base = (id - 1) * dim
        local dst_base = (i - 1) * dim
        if pe then
            local row = pe[i + offset]
            for j = 1, dim do
                dst[dst_base + j] = src[src_base + j] * escale + row[j] * pscale
            end
        else
            for j = 1, dim do
                dst[dst_base + j] = src[src_base + j] * escale
            end
        end
    end

    local embed_ref = self.embed
    local out = Tensor.new(out_data, {embed_ref}, "embed_lookup")
    out.backward_fn = function()
        local eg, og = embed_ref.grad.data, out.grad.data
        for i = 1, seq_len do
            local id = token_ids[i]
            local src_base = (id - 1) * dim
            local dst_base = (i - 1) * dim
            for j = 1, dim do
                eg[src_base + j] = eg[src_base + j] + og[dst_base + j] * escale
            end
        end
        -- the positional term is a constant: no gradient path.
    end
    return out
end

-- opts.offset: absolute position of token 1 (KV-cache / sliding window).
function Transformer:forward(token_ids, opts)
    local offset = (opts and opts.offset) or 0
    local x = self:embed_tokens(token_ids, offset)

    local block_opts = (offset ~= 0) and { offset = offset } or nil
    for i = 1, self.layers do
        x = self.blocks[i]:forward(x, block_opts)
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

function Transformer:num_parameters()
    local n = 0
    for _, p in ipairs(self:parameters()) do
        n = n + p.data.rows * p.data.cols
    end
    return n
end

return Transformer
