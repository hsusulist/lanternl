local matrix = require("matrix")
local Tensor = require("tensor2")

local Transformer = {}
Transformer.__index = Transformer

local sqrt, exp, cos, sin, huge = math.sqrt, math.exp, math.cos, math.sin, math.huge
local sformat = string.format
local alloc = matrix.alloc

local function uniform(m, a)
    local d = m.data
    for k = 1, m.rows * m.cols do
        d[k] = (math.random() * 2 - 1) * a
    end
    return m
end

local function linear(fin, fout)
    local m = alloc(fin, fout)
    return Tensor.param(uniform(m, sqrt(6 / (fin + fout))))
end

local function ones_row(n)
    local m = alloc(1, n)
    for k = 1, n do m.data[k] = 1 end
    return Tensor.param(m)
end

local function build_rope(base, hd, maxpos)
    local npairs = math.floor(hd / 2)
    local inv = {}
    for k = 1, npairs do
        inv[k] = 1 / (base ^ ((2 * (k - 1)) / hd))
    end
    local c, s = {}, {}
    for p = 0, maxpos do
        local cr, sr = {}, {}
        for k = 1, npairs do
            local ang = p * inv[k]
            cr[k] = cos(ang)
            sr[k] = sin(ang)
        end
        c[p + 1] = cr
        s[p + 1] = sr
    end
    return { cos = c, sin = s, inv = inv, npairs = npairs, maxpos = maxpos, base = base, hd = hd }
end

local function rope_extend(rp, maxpos)
    if maxpos <= rp.maxpos then return rp end
    for p = rp.maxpos + 1, maxpos do
        local cr, sr = {}, {}
        for k = 1, rp.npairs do
            local ang = p * rp.inv[k]
            cr[k] = cos(ang)
            sr[k] = sin(ang)
        end
        rp.cos[p + 1] = cr
        rp.sin[p + 1] = sr
    end
    rp.maxpos = maxpos
    return rp
end

local function sinusoidal_table(dim, maxpos)
    local m = alloc(maxpos + 1, dim)
    local d = m.data
    for p = 0, maxpos do
        local base = p * dim
        for j = 1, dim do
            local i = math.floor((j - 1) / 2)
            local ang = p / (10000 ^ ((2 * i) / dim))
            if (j % 2) == 1 then
                d[base + j] = sin(ang)
            else
                d[base + j] = cos(ang)
            end
        end
    end
    return m
end

function Transformer.new(cfg)
    cfg = cfg or {}
    local self = setmetatable({}, Transformer)

    self.vocab = cfg.vocab or 256
    self.dim = cfg.dim or 64
    self.nlayers = cfg.layers or 2
    self.heads = cfg.heads or 4
    self.ffn = cfg.ffn or 128
    self.causal = cfg.causal ~= false
    self.pos = string.lower(cfg.pos or "rope")
    self.rope_base = cfg.rope_base or 10000
    self.max_seq = cfg.max_seq or 512
    self.tie = cfg.tie ~= false
    self.eps = cfg.eps or 1e-5
    self.dropout = cfg.dropout or 0

    if self.dim % self.heads ~= 0 then
        error(sformat("Transformer: dim (%d) must be divisible by heads (%d)",
            self.dim, self.heads), 2)
    end
    self.hd = self.dim / self.heads
    self.scale = 1 / sqrt(self.hd)

    if self.pos == "rotary" then self.pos = "rope" end
    if self.pos == "sin" then self.pos = "sinusoidal" end
    if self.pos == "abs" then self.pos = "absolute" end
    if self.pos == "off" then self.pos = "none" end

    if self.pos == "rope" then
        if self.hd < 2 then
            error("Transformer: RoPE needs dim/heads >= 2", 2)
        end
        self.rope = build_rope(self.rope_base, self.hd, self.max_seq)
    elseif self.pos == "sinusoidal" then
        self.pe = sinusoidal_table(self.dim, self.max_seq)
    elseif self.pos == "absolute" then
        self.pe_param = Tensor.param(uniform(alloc(self.max_seq + 1, self.dim), 0.02))
    end

    local emb = alloc(self.vocab, self.dim)
    self.emb = Tensor.param(uniform(emb, 0.02))

    self.layers = {}
    for i = 1, self.nlayers do
        self.layers[i] = {
            ln1 = ones_row(self.dim),
            wq = linear(self.dim, self.dim),
            wk = linear(self.dim, self.dim),
            wv = linear(self.dim, self.dim),
            wo = linear(self.dim, self.dim),
            ln2 = ones_row(self.dim),
            w1 = linear(self.dim, self.ffn),
            w3 = linear(self.dim, self.ffn),
            w2 = linear(self.ffn, self.dim),
        }
    end
    self.lnf = ones_row(self.dim)
    if not self.tie then
        self.head = linear(self.dim, self.vocab)
    end

    self._params = nil
    return self
end

function Transformer:parameters()
    if self._params then return self._params end
    local p = { self.emb }
    if self.pe_param then p[#p + 1] = self.pe_param end
    for i = 1, self.nlayers do
        local L = self.layers[i]
        p[#p + 1] = L.ln1; p[#p + 1] = L.wq; p[#p + 1] = L.wk
        p[#p + 1] = L.wv;  p[#p + 1] = L.wo; p[#p + 1] = L.ln2
        p[#p + 1] = L.w1;  p[#p + 1] = L.w3; p[#p + 1] = L.w2
    end
    p[#p + 1] = self.lnf
    if self.head then p[#p + 1] = self.head end
    self._params = p
    return p
end

function Transformer:rope_rows(n, offset)
    offset = offset or 0
    rope_extend(self.rope, n + offset)
    local c, s = {}, {}
    for i = 1, n do
        c[i] = self.rope.cos[i + offset]
        s[i] = self.rope.sin[i + offset]
    end
    return c, s
end

function Transformer:_pos_embed(x, T, offset)
    offset = offset or 0
    if self.pos == "sinusoidal" then
        local pe = alloc(T, self.dim)
        local pd, sd = pe.data, self.pe.data
        for i = 1, T do
            local src = (i - 1 + offset) * self.dim
            local dst = (i - 1) * self.dim
            for j = 1, self.dim do pd[dst + j] = sd[src + j] end
        end
        return Tensor.add(x, Tensor.from_matrix(pe))
    elseif self.pos == "absolute" then
        local ids = {}
        for i = 1, T do ids[i] = i + offset end
        return Tensor.add(x, Tensor.embed(self.pe_param, ids))
    end
    return x
end

function Transformer:_attention(h, L, T)
    local q = Tensor.matmul(h, L.wq)
    local k = Tensor.matmul(h, L.wk)
    local v = Tensor.matmul(h, L.wv)

    local cr, sr
    if self.pos == "rope" then
        cr, sr = self:rope_rows(T, 0)
    end

    local hd = self.hd
    local outs = {}
    for hh = 1, self.heads do
        local s0 = (hh - 1) * hd + 1
        local s1 = hh * hd
        local qh = Tensor.slice_cols(q, s0, s1)
        local kh = Tensor.slice_cols(k, s0, s1)
        local vh = Tensor.slice_cols(v, s0, s1)
        if cr then
            qh = Tensor.rope(qh, cr, sr)
            kh = Tensor.rope(kh, cr, sr)
        end
        local sc = Tensor.scale(Tensor.matmul_nt(qh, kh), self.scale)
        local pr = Tensor.softmax_rows(sc, self.causal)
        pr = Tensor.dropout(pr, self.dropout)
        outs[hh] = Tensor.matmul(pr, vh)
    end

    local ctx = self.heads == 1 and outs[1] or Tensor.concat_cols(outs)
    return Tensor.matmul(ctx, L.wo)
end

function Transformer:forward(ids)
    if type(ids) ~= "table" or #ids == 0 then
        error("Transformer:forward: need a non-empty table of token ids", 2)
    end
    local T = #ids
    if T > self.max_seq then
        error(sformat("Transformer:forward: sequence of %d tokens exceeds max_seq %d",
            T, self.max_seq), 2)
    end

    local x = Tensor.embed(self.emb, ids)
    x = self:_pos_embed(x, T, 0)

    for i = 1, self.nlayers do
        local L = self.layers[i]
        local h = Tensor.rmsnorm(x, L.ln1, self.eps)
        x = Tensor.add(x, Tensor.dropout(self:_attention(h, L, T), self.dropout))
        local h2 = Tensor.rmsnorm(x, L.ln2, self.eps)
        local g = Tensor.silu(Tensor.matmul(h2, L.w1))
        local u = Tensor.matmul(h2, L.w3)
        local f = Tensor.matmul(Tensor.mul(g, u), L.w2)
        x = Tensor.add(x, Tensor.dropout(f, self.dropout))
    end

    local xf = Tensor.rmsnorm(x, self.lnf, self.eps)
    if self.tie then
        return Tensor.matmul_nt(xf, self.emb)
    end
    return Tensor.matmul(xf, self.head)
end

local function vecmat(x, W, fin, fout, out)
    for j = 1, fout do out[j] = 0 end
    for i = 1, fin do
        local xv = x[i]
        if xv ~= 0 then
            local base = (i - 1) * fout
            for j = 1, fout do
                out[j] = out[j] + xv * W[base + j]
            end
        end
    end
    return out
end

local function rmsnorm_vec(x, w, n, eps, out)
    local ss = 0
    for j = 1, n do
        local v = x[j]
        ss = ss + v * v
    end
    local r = 1 / sqrt(ss / n + eps)
    for j = 1, n do out[j] = x[j] * r * w[j] end
    return out
end

function Transformer:new_state()
    local st = { n = 0, k = {}, v = {}, cap = self.max_seq }
    for i = 1, self.nlayers do
        st.k[i] = {}
        st.v[i] = {}
    end
    st.buf = {
        x = {}, h = {}, q = {}, kk = {}, vv = {}, o = {},
        ctx = {}, g = {}, u = {}, f = {}, att = {}, logits = {},
    }
    return st
end

function Transformer:step(id, state)
    local dim, hd, heads = self.dim, self.hd, self.heads
    local pos = state.n
    if pos >= self.max_seq then
        error(sformat("Transformer:step: context is full (%d tokens)", self.max_seq), 2)
    end
    if type(id) ~= "number" or id < 1 or id > self.vocab then
        error("Transformer:step: token id out of range: " .. tostring(id), 2)
    end

    local B = state.buf
    local x, h, q, kk, vv, o = B.x, B.h, B.q, B.kk, B.vv, B.o
    local ctxv, g, u, f, att = B.ctx, B.g, B.u, B.f, B.att

    local ed = self.emb.data.data
    local esrc = (id - 1) * dim
    for j = 1, dim do x[j] = ed[esrc + j] end

    if self.pos == "sinusoidal" then
        local pd = self.pe.data
        local base = pos * dim
        for j = 1, dim do x[j] = x[j] + pd[base + j] end
    elseif self.pos == "absolute" then
        local pd = self.pe_param.data.data
        local base = pos * dim
        for j = 1, dim do x[j] = x[j] + pd[base + j] end
    end

    local cr, sr
    if self.pos == "rope" then
        rope_extend(self.rope, pos)
        cr = self.rope.cos[pos + 1]
        sr = self.rope.sin[pos + 1]
    end
    local npairs = cr and math.floor(hd / 2) or 0

    for l = 1, self.nlayers do
        local L = self.layers[l]
        rmsnorm_vec(x, L.ln1.data.data, dim, self.eps, h)
        vecmat(h, L.wq.data.data, dim, dim, q)
        vecmat(h, L.wk.data.data, dim, dim, kk)
        vecmat(h, L.wv.data.data, dim, dim, vv)

        if cr then
            for hh = 1, heads do
                local off = (hh - 1) * hd
                for p = 1, npairs do
                    local i0 = off + 2 * p - 1
                    local i1 = i0 + 1
                    local c1, s1 = cr[p], sr[p]
                    local a0, a1 = q[i0], q[i1]
                    q[i0] = a0 * c1 - a1 * s1
                    q[i1] = a0 * s1 + a1 * c1
                    local b0, b1 = kk[i0], kk[i1]
                    kk[i0] = b0 * c1 - b1 * s1
                    kk[i1] = b0 * s1 + b1 * c1
                end
            end
        end

        local kc, vc = state.k[l], state.v[l]
        local cbase = pos * dim
        for j = 1, dim do
            kc[cbase + j] = kk[j]
            vc[cbase + j] = vv[j]
        end

        local n = pos + 1
        for hh = 1, heads do
            local off = (hh - 1) * hd
            local max_s = -huge
            for t = 1, n do
                local kb = (t - 1) * dim + off
                local s = 0
                for j = 1, hd do
                    s = s + q[off + j] * kc[kb + j]
                end
                s = s * self.scale
                att[t] = s
                if s > max_s then max_s = s end
            end
            local sum = 0
            for t = 1, n do
                local e = exp(att[t] - max_s)
                att[t] = e
                sum = sum + e
            end
            local inv = 1 / sum
            for j = 1, hd do ctxv[off + j] = 0 end
            for t = 1, n do
                local w = att[t] * inv
                if w ~= 0 then
                    local vb = (t - 1) * dim + off
                    for j = 1, hd do
                        ctxv[off + j] = ctxv[off + j] + w * vc[vb + j]
                    end
                end
            end
        end

        vecmat(ctxv, L.wo.data.data, dim, dim, o)
        for j = 1, dim do x[j] = x[j] + o[j] end

        rmsnorm_vec(x, L.ln2.data.data, dim, self.eps, h)
        vecmat(h, L.w1.data.data, dim, self.ffn, g)
        vecmat(h, L.w3.data.data, dim, self.ffn, u)
        for j = 1, self.ffn do
            local vgg = g[j]
            local sg
            if vgg >= 0 then
                sg = 1 / (1 + exp(-vgg))
            else
                local z = exp(vgg)
                sg = z / (1 + z)
            end
            g[j] = vgg * sg * u[j]
        end
        vecmat(g, L.w2.data.data, self.ffn, dim, f)
        for j = 1, dim do x[j] = x[j] + f[j] end
    end

    rmsnorm_vec(x, self.lnf.data.data, dim, self.eps, h)
    local logits = B.logits
    if self.tie then
        local wd = self.emb.data.data
        for r = 1, self.vocab do
            local base = (r - 1) * dim
            local s = 0
            for j = 1, dim do s = s + h[j] * wd[base + j] end
            logits[r] = s
        end
    else
        vecmat(h, self.head.data.data, dim, self.vocab, logits)
    end

    state.n = pos + 1
    return logits
end

function Transformer:prefill(ids, state)
    local logits
    for i = 1, #ids do
        logits = self:step(ids[i], state)
    end
    return logits
end

return Transformer
