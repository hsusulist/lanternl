-- =====================================================================
--  gpu_transformer.lua — fully GPU-resident Transformer training step
--
--  One training step  =  token ids in  ->  loss out  ->  weights updated
--  in place on the GPU.  The host boundary is crossed exactly twice:
--
--      1. upload  N int32 input ids + N int32 targets   (H2D, ~8 KB)
--      2. download 1 float loss  (+ N int32 argmax if accuracy is on)
--
--  Everything between those two points -- embedding lookup, RoPE, causal
--  multi-head attention, RMSNorm, SwiGLU FFN, tied/untied head, fused
--  cross-entropy, the entire backward pass, global gradient clipping and
--  the AdamW update -- is queued into luaTL.Program objects and executed
--  with a handful of FFI transitions (or ONE, with CUDA graph replay).
--
--  Layout
--  ------
--    N  = B * T          rows; sequence b occupies rows b*T .. b*T+T-1
--    D  = dim, H = heads, hd = D/H, F = ffn hidden, V = vocab
--    qkv[l]    : [N, 3D]        Q = cols 0..D-1, K = D..2D-1, V = 2D..3D-1
--    scores[l] : [B*H*T, T]     row ((b*H+h)*T + t);  r % T == t  =>  the
--                               group-causal softmax masks correctly and
--                               sequences cannot see each other.
--
--  Padding contract
--  ----------------
--    Shapes are STATIC.  A short window or a short final batch is padded
--    with id 0 and target -1.  luaTL_ce_kernel writes loss 0 AND
--    dlogits 0 for those rows, so their contribution to every gradient is
--    exactly zero.  grad_scale and the loss-sum scale are patched in the
--    command structs to 1/valid_tokens, so the reported loss and the
--    gradient magnitude are exact, not approximate.
-- =====================================================================

local ffi = require("ffi")

local adapter = require("luatl_adapter")

local GT = {}
GT.__index = GT

local Model = {}
Model.__index = Model
GT.Model = Model

local sqrt, log, cos, sin, pi = math.sqrt, math.log, math.cos, math.sin, math.pi
local floor, min, max, huge = math.floor, math.min, math.max, math.huge
local sformat = string.format

-- ---------------------------------------------------------------------
--  Bindings (resolved lazily so that requiring this file on a CPU-only
--  box does not explode; construction is what fails loudly).
-- ---------------------------------------------------------------------
local luaTL, C, F32

local function bind()
    if luaTL then return end
    if not adapter.available then
        error("gpu_transformer: luaTL is not available (" ..
              tostring(adapter.error) .. ").\n" ..
              "  Build the CUDA backend first:\n" ..
              "    nvcc -O3 -std=c++14 --shared -Xcompiler -fPIC \\\n" ..
              "         -gencode arch=compute_86,code=sm_86 --use_fast_math \\\n" ..
              "         luaTL_train.cu -o luaTL.so", 3)
    end
    luaTL = adapter.luaTL
    C     = luaTL.C
    F32   = luaTL.F32
end

GT.available = adapter.available
GT.adapter   = adapter

-- ---------------------------------------------------------------------
--  Small helpers
-- ---------------------------------------------------------------------

--- Byte-offset pointer into a tensor's storage, in ELEMENTS (f32).
local function poff(t, elems)
    return ffi.cast("char*", t.data) + elems * 4
end

--- Stride view: the six numbers gemm_ex needs, plus a batch stride.
local function sv(ptr, rows, cols, rs, cs, bs)
    return { data = ptr, rows = rows, cols = cols,
             rs = rs, cs = cs, bs = bs or 0, dtype = F32 }
end

--- Grab the command struct that was just queued, so we can patch scalars
--- (grad_scale, lr, step) later without re-queueing the whole program.
local function last_op(p)
    return p.h.ops[p.h.count - 1]
end

local function normal_fill(t, std)
    local n = t.rows * t.cols
    local buf = ffi.new("float[?]", n)
    local i = 0
    while i < n do
        local u1 = math.random()
        local u2 = math.random()
        if u1 < 1e-12 then u1 = 1e-12 end
        local r = sqrt(-2.0 * log(u1)) * std
        buf[i] = r * cos(2 * pi * u2); i = i + 1
        if i < n then buf[i] = r * sin(2 * pi * u2); i = i + 1 end
    end
    t:upload(buf, n)
end

-- =====================================================================
--  Construction
-- =====================================================================

local function need(cfg, key)
    local v = cfg[key]
    if type(v) ~= "number" or v ~= floor(v) or v < 1 then
        error(sformat("gpu_transformer: '%s' must be a positive integer, got %s",
                      key, tostring(v)), 3)
    end
    return v
end

function GT.new(cfg)
    bind()
    cfg = cfg or {}

    local self = setmetatable({}, Model)

    self.vocab   = need(cfg, "vocab")
    self.dim     = need(cfg, "dim")
    self.layers  = need(cfg, "layers")
    self.heads   = need(cfg, "heads")
    self.ffn     = need(cfg, "ffn")
    self.T       = need(cfg, "max_seq")
    self.B       = need(cfg, "batch_size")

    if self.dim % self.heads ~= 0 then
        error(sformat("gpu_transformer: dim (%d) must be divisible by heads (%d)",
                      self.dim, self.heads), 2)
    end
    self.hd = self.dim / self.heads
    if self.hd % 2 ~= 0 then
        error(sformat("gpu_transformer: head_dim (dim/heads = %d) must be even for RoPE",
                      self.hd), 2)
    end

    local pos = string.lower(tostring(cfg.pos or "rope"))
    if pos == "rope" or pos == "rotary" then
        self.use_rope = true
    elseif pos == "none" or pos == "off" then
        self.use_rope = false
    else
        error(sformat("gpu_transformer: pos = '%s' is not supported by the GPU backend.\n" ..
              "  Use pos = 'rope' or pos = 'none', or switch to backend = 'cpu'.", pos), 2)
    end

    if cfg.dropout and cfg.dropout > 0 then
        error(sformat("gpu_transformer: dropout = %g is not supported by the GPU backend " ..
              "(no device RNG kernel exists yet).\n" ..
              "  Set dropout = 0, or use backend = 'cpu' if you need it.",
              cfg.dropout), 2)
    end

    self.causal      = (cfg.causal ~= false)
    self.tie         = (cfg.tie ~= false)
    self.eps         = cfg.eps or 1e-5
    self.rope_base   = cfg.rope_base or 10000
    self.clip_norm   = cfg.clip_norm or 0
    self.weight_decay= cfg.weight_decay or 0.0
    self.beta1       = cfg.beta1 or 0.9
    self.beta2       = cfg.beta2 or 0.95
    self.adam_eps    = cfg.adam_eps or 1e-8
    self.want_acc    = (cfg.want_acc ~= false)
    self.use_graph   = (cfg.graph == true)
    self.timing      = (cfg.timing == true)

    self.N          = self.B * self.T
    self.attn_scale = 1.0 / sqrt(self.hd)

    self.params = {}
    self._keep  = {}

    self:_alloc()
    self:_build_programs()

    return self
end

-- ---------------------------------------------------------------------
--  Allocation.  Every tensor below lives for the lifetime of the model.
--  Handles (adapter.alloc / adapter.pin) own the storage and anchor it
--  against the GC; `self._keep` anchors the non-owning wrap views.
-- ---------------------------------------------------------------------
function Model:_alloc()
    local B, T, D, H, hd = self.B, self.T, self.dim, self.heads, self.hd
    local F, V, L, N     = self.ffn, self.vocab, self.layers, self.N
    local D3, F2         = 3 * D, 2 * F
    local keep           = self._keep

    local function A(rows, cols)                 -- zeroed activation buffer
        local h = adapter.alloc(rows, cols)
        keep[#keep + 1] = h
        return h.t
    end
    local function W(ptr, rows, cols)            -- non-owning column view
        local t = luaTL.wrap(ptr, rows, cols, F32)
        keep[#keep + 1] = t
        return t
    end

    local function param(name, rows, cols, init, decay)
        local h = adapter.alloc(rows, cols)
        if init == "ones" then
            h.t:fill(1.0)
        elseif type(init) == "number" then
            normal_fill(h.t, init)
        end
        local p = {
            name = name, rows = rows, cols = cols,
            w = h,
            g = adapter.alloc(rows, cols),
            m = adapter.alloc(rows, cols),
            v = adapter.alloc(rows, cols),
            decay = decay and true or false,
            nelem = rows * cols,
        }
        self.params[#self.params + 1] = p
        return p
    end

    -- ---- parameters -------------------------------------------------
    -- GPT-2 style init: 0.02 everywhere, residual-output projections
    -- scaled by 1/sqrt(2L) so the residual stream does not blow up with
    -- depth.  No weight decay on 1-D norm gains.
    local std  = 0.02
    local rstd = std / sqrt(2 * L)

    self.emb = param("tok_emb", V, D, std, true)
    self.lyr = {}
    for l = 1, L do
        local ly = {}
        ly.ln1  = param(sformat("l%d.ln1",  l), 1, D,   "ones", false)
        ly.wqkv = param(sformat("l%d.wqkv", l), D, D3,  std,    true)
        ly.wo   = param(sformat("l%d.wo",   l), D, D,   rstd,   true)
        ly.ln2  = param(sformat("l%d.ln2",  l), 1, D,   "ones", false)
        ly.w13  = param(sformat("l%d.w13",  l), D, F2,  std,    true)
        ly.w2   = param(sformat("l%d.w2",   l), F, D,   rstd,   true)
        self.lyr[l] = ly
    end
    self.lnf = param("lnf", 1, D, "ones", false)
    if not self.tie then
        self.wout = param("head", D, V, std, true)
    end

    -- ---- index buffers ----------------------------------------------
    self.ids = A(N, 1)     -- int32 storage in an f32-sized tensor
    self.tgt = A(N, 1)
    self.pos = A(N, 1)

    self.h_ids  = ffi.new("int32_t[?]", N)
    self.h_tgt  = ffi.new("int32_t[?]", N)
    self.h_pred = ffi.new("int32_t[?]", N)
    self.h_pos  = ffi.new("int32_t[?]", N)
    for r = 0, N - 1 do self.h_pos[r] = r % T end
    luaTL.check(C.luaTL_upload_i32(self.pos.data, self.h_pos, N), "pos.upload")

    -- ---- residual stream + per-layer saved activations ---------------
    self.res  = { [0] = A(N, D) }      -- res[l]  : output of layer l
    self.mid  = {}                     -- mid[l]  : after attention residual
    self.h1, self.qkv, self.sc, self.attn = {}, {}, {}, {}
    self.h2, self.h13, self.hs            = {}, {}, {}
    self.qv, self.kv, self.h13a, self.h13b = {}, {}, {}, {}

    for l = 1, L do
        self.res[l]  = A(N, D)
        self.mid[l]  = A(N, D)
        self.h1[l]   = A(N, D)
        self.qkv[l]  = A(N, D3)
        self.sc[l]   = A(B * H * T, T)
        self.attn[l] = A(N, D)
        self.h2[l]   = A(N, D)
        self.h13[l]  = A(N, F2)
        self.hs[l]   = A(N, F)
        -- RoPE operates in place on the Q and K column slices
        self.qv[l]   = W(poff(self.qkv[l], 0), N, D)
        self.kv[l]   = W(poff(self.qkv[l], D), N, D)
        -- SwiGLU gate/up halves of the fused [N, 2F] projection
        self.h13a[l] = W(poff(self.h13[l], 0), N, F)
        self.h13b[l] = W(poff(self.h13[l], F), N, F)
    end

    self.hf     = A(N, D)
    self.logits = A(N, V)          -- dlogits aliases this (saves N*V floats)
    self.losses = A(N, 1)
    self.pred   = A(N, 1)
    self.lossacc= A(1, 1)
    self.nsq    = A(1, 1)

    -- ---- backward scratch (reused across layers) ---------------------
    self.dres   = A(N, D)
    self.dbr    = A(N, D)
    self.dhf    = A(N, D)
    self.dh1    = A(N, D)
    self.dqkv   = A(N, D3)
    self.dsc    = A(B * H * T, T)
    self.dattn  = A(N, D)
    self.dh2    = A(N, D)
    self.dh13   = A(N, F2)
    self.dhs    = A(N, F)
    self.dqv    = W(poff(self.dqkv, 0), N, D)
    self.dkv    = W(poff(self.dqkv, D), N, D)
    self.dh13a  = W(poff(self.dh13, 0), N, F)
    self.dh13b  = W(poff(self.dh13, F), N, F)

    self.f1 = ffi.new("float[1]")
end

-- =====================================================================
--  Program construction
-- =====================================================================

--- Forward pass.  `grad` selects whether cross-entropy also writes
--- dlogits (in place over the logits) or leaves the logits intact so
--- that :generate() can read them.
function Model:_queue_forward(p, grad)
    local B, T, D, H, hd = self.B, self.T, self.dim, self.heads, self.hd
    local F, V, L, N     = self.ffn, self.vocab, self.layers, self.N
    local D3, F2         = 3 * D, 2 * F
    local eps            = self.eps
    local TT             = T * T
    local HTT            = H * TT

    p:zero(self.lossacc)
    p:embed(self.ids, self.emb.w.t, self.res[0])

    for l = 1, L do
        local ly   = self.lyr[l]
        local xin  = self.res[l - 1]
        local mid  = self.mid[l]
        local xout = self.res[l]
        local qkv, sc, at = self.qkv[l], self.sc[l], self.attn[l]

        -- pre-attention norm + fused QKV projection
        p:rmsnorm(xin, ly.ln1.w.t, self.h1[l], eps)
        p:gemm(self.h1[l], ly.wqkv.w.t, qkv)

        if self.use_rope then
            local ro = { heads = H, head_dim = hd, row_stride = D3,
                         pos_ids = self.pos, theta = self.rope_base }
            p:rope(self.qv[l], ro)
            p:rope(self.kv[l], ro)
        end

        -- S = Q Kt   (batched over heads, looped over sequences)
        for b = 0, B - 1 do
            local qb = b * T * D3
            p:gemm(sv(poff(qkv, qb),         T, hd, D3, 1,  hd),   -- Q  [T,hd]
                   sv(poff(qkv, qb + D),     hd, T,  1, D3, hd),   -- Kt [hd,T]
                   sv(poff(sc,  b * HTT),    T,  T,  T, 1,  TT),   -- S  [T,T]
                   { batch = H })
        end

        -- one launch masks + softmaxes every (sequence, head) row block
        p:softmax(sc, sc, { causal = self.causal, group = T,
                            scale = self.attn_scale })

        -- ctx = P V
        for b = 0, B - 1 do
            local qb = b * T * D3
            p:gemm(sv(poff(sc,  b * HTT),    T, T,  T,  1, TT),
                   sv(poff(qkv, qb + 2 * D), T, hd, D3, 1, hd),
                   sv(poff(at,  b * T * D),  T, hd, D,  1, hd),
                   { batch = H })
        end

        -- residual: mid = xin + ctx @ Wo   (add fused into the epilogue)
        p:copy(xin, mid)
        p:gemm(at, ly.wo.w.t, mid, { beta = 1.0 })

        -- SwiGLU feed-forward
        p:rmsnorm(mid, ly.ln2.w.t, self.h2[l], eps)
        p:gemm(self.h2[l], ly.w13.w.t, self.h13[l])
        p:swiglu(self.h13a[l], self.h13b[l], self.hs[l],
                 { lda = F2, ldb = F2, ldo = F })
        p:copy(mid, xout)
        p:gemm(self.hs[l], ly.w2.w.t, xout, { beta = 1.0 })
    end

    -- final norm + output head
    p:rmsnorm(self.res[L], self.lnf.w.t, self.hf, eps)
    if self.tie then
        p:gemm(self.hf, self.emb.w.t:t(), self.logits)   -- [N,D] @ [D,V]
    else
        p:gemm(self.hf, self.wout.w.t, self.logits)
    end

    -- argmax BEFORE cross-entropy, because dlogits aliases logits
    if self.want_acc then p:argmax(self.logits, self.pred) end

    p:cross_entropy(self.logits, self.tgt, self.losses,
                    grad and self.logits or nil,
                    { ignore_index = -1, grad_scale = 1.0 / self.N })
    self._ce_ops[#self._ce_ops + 1] = last_op(p)

    p:sum(self.losses, self.lossacc, 1.0 / self.N)
    self._sum_ops[#self._sum_ops + 1] = last_op(p)
end

function Model:_queue_backward(p)
    local B, T, D, H, hd = self.B, self.T, self.dim, self.heads, self.hd
    local F, L, N        = self.ffn, self.layers, self.N
    local D3, F2         = 3 * D, 2 * F
    local eps            = self.eps
    local TT             = T * T
    local HTT            = H * TT
    local dl             = self.logits          -- dlogits, written in place

    -- ---- output head -------------------------------------------------
    if self.tie then
        -- d(tok_emb) += dlogits^T @ hf     ([V,N] @ [N,D] -> [V,D])
        p:gemm(dl:t(), self.hf, self.emb.g.t, { beta = 1.0 })
        -- dhf = dlogits @ tok_emb          ([N,V] @ [V,D] -> [N,D])
        p:gemm(dl, self.emb.w.t, self.dhf)
    else
        p:dw(self.hf, dl, self.wout.g.t)
        p:dx(dl, self.wout.w.t, self.dhf)
    end
    p:rmsnorm_bwd(self.res[L], self.lnf.w.t, self.dhf, self.dres,
                  self.lnf.g.t, eps)

    -- ---- layers, top down --------------------------------------------
    for l = L, 1, -1 do
        local ly  = self.lyr[l]
        local xin = self.res[l - 1]
        local mid = self.mid[l]
        local qkv, sc, at = self.qkv[l], self.sc[l], self.attn[l]

        -- ---- feed-forward ----
        p:dx(self.dres, ly.w2.w.t, self.dhs)          -- dhs  = dres @ W2^T
        p:dw(self.hs[l], self.dres, ly.w2.g.t)        -- dW2 += hs^T @ dres
        p:swiglu_bwd(self.h13a[l], self.h13b[l], self.dhs,
                     self.dh13a, self.dh13b,
                     { lda = F2, ldb = F2, ldo = F })
        p:dw(self.h2[l], self.dh13, ly.w13.g.t)
        p:dx(self.dh13, ly.w13.w.t, self.dh2)
        p:rmsnorm_bwd(mid, ly.ln2.w.t, self.dh2, self.dbr, ly.ln2.g.t, eps)
        p:axpy(self.dres, self.dbr, 1.0)              -- residual join

        -- ---- attention ----
        p:dx(self.dres, ly.wo.w.t, self.dattn)
        p:dw(at, self.dres, ly.wo.g.t)

        for b = 0, B - 1 do
            local qb = b * T * D3
            local dA = sv(poff(self.dattn, b * T * D), T, hd, D, 1, hd)
            -- dP = dctx @ V^T
            p:gemm(dA,
                   sv(poff(qkv, qb + 2 * D), hd, T, 1, D3, hd),
                   sv(poff(self.dsc, b * HTT), T, T, T, 1, TT),
                   { batch = H })
            -- dV = P^T @ dctx
            p:gemm(sv(poff(sc, b * HTT), T, T, 1, T, TT),
                   dA,
                   sv(poff(self.dqkv, qb + 2 * D), T, hd, D3, 1, hd),
                   { batch = H })
        end

        -- dS = softmax'(P) . dP   (aliases dP; safe, the row reduction
        -- completes behind a __syncthreads before any store)
        p:softmax_bwd(sc, self.dsc, self.dsc,
                      { causal = self.causal, group = T,
                        scale = self.attn_scale })

        for b = 0, B - 1 do
            local qb = b * T * D3
            -- dQ = dS @ K
            p:gemm(sv(poff(self.dsc, b * HTT), T, T, T, 1, TT),
                   sv(poff(qkv, qb + D), T, hd, D3, 1, hd),
                   sv(poff(self.dqkv, qb), T, hd, D3, 1, hd),
                   { batch = H })
            -- dK = dS^T @ Q
            p:gemm(sv(poff(self.dsc, b * HTT), T, T, 1, T, TT),
                   sv(poff(qkv, qb), T, hd, D3, 1, hd),
                   sv(poff(self.dqkv, qb + D), T, hd, D3, 1, hd),
                   { batch = H })
        end

        if self.use_rope then
            local ro = { heads = H, head_dim = hd, row_stride = D3,
                         pos_ids = self.pos, theta = self.rope_base,
                         inverse = true }
            p:rope(self.dqv, ro)
            p:rope(self.dkv, ro)
        end

        p:dw(self.h1[l], self.dqkv, ly.wqkv.g.t)
        p:dx(self.dqkv, ly.wqkv.w.t, self.dh1)
        p:rmsnorm_bwd(xin, ly.ln1.w.t, self.dh1, self.dbr, ly.ln1.g.t, eps)
        p:axpy(self.dres, self.dbr, 1.0)
    end

    -- ---- embedding scatter-add (accumulates on top of the tied head) --
    p:embed_bwd(self.ids, self.dres, self.emb.g.t, 1.0)
end

function Model:_queue_zero(p)
    for i = 1, #self.params do p:zero(self.params[i].g.t) end
end

function Model:_queue_opt(p)
    local clip = self.clip_norm or 0
    if clip > 0 then
        p:zero(self.nsq)
        for i = 1, #self.params do p:l2(self.params[i].g.t, self.nsq) end
        for i = 1, #self.params do
            p:clip(self.params[i].g.t, self.nsq, clip)
        end
    end
    self._adam_ops = {}
    for i = 1, #self.params do
        local pp = self.params[i]
        p:adamw(pp.w.t, pp.g.t, pp.m.t, pp.v.t, nil, {
            lr    = 1e-3,                 -- patched every step
            beta1 = self.beta1,
            beta2 = self.beta2,
            eps   = self.adam_eps,
            wd    = pp.decay and self.weight_decay or 0.0,
            step  = 1,                    -- patched every step
            clip  = 0.0,
        })
        self._adam_ops[i] = last_op(p)
    end
end

function Model:_build_programs()
    local L, B = self.layers, self.B
    local nparam = #self.params

    -- generous, computed capacity: fwd ~ (10 + 2B)/layer, bwd ~ (16 + 4B)
    local cap = 256 + L * (48 + 8 * B) + nparam * 8

    self._ce_ops, self._sum_ops = {}, {}

    self.p_zero = luaTL.Program(nparam + 8, false)
    self:_queue_zero(self.p_zero)

    self.p_fwd = luaTL.Program(cap, self.timing)
    self:_queue_forward(self.p_fwd, false)

    self.p_step = luaTL.Program(cap, self.timing)
    self:_queue_forward(self.p_step, true)
    self:_queue_backward(self.p_step)

    self.p_opt = luaTL.Program(nparam * 4 + 16, false)
    self:_queue_opt(self.p_opt)

    self.graph_ready = false
    self._cap_scale  = nil
end

-- =====================================================================
--  Per-step scalar patching (grad_scale / loss scale / lr / adam step)
-- =====================================================================
function Model:_set_valid(valid)
    if valid < 1 then valid = 1 end
    if self._valid == valid then return end
    self._valid = valid
    local s = 1.0 / valid
    for i = 1, #self._ce_ops  do self._ce_ops[i].f0  = s end
    for i = 1, #self._sum_ops do self._sum_ops[i].f0 = s end
end

function Model:_set_hparams(lr, step)
    local ops = self._adam_ops
    for i = 1, #ops do
        ops[i].f0 = lr
        ops[i].i0 = step
    end
end

-- =====================================================================
--  Host <-> device: the ONLY two places that touch the CPU
-- =====================================================================

--- Pack up to B windows into the static [B, T] id/target block.
--- Returns the number of valid (non-padding) target positions.
function Model:_upload(windows, nwin)
    local B, T, N = self.B, self.T, self.N
    local ib, tb  = self.h_ids, self.h_tgt
    local V       = self.vocab
    local valid   = 0

    for b = 0, B - 1 do
        local base = b * T
        local w    = (b < (nwin or B)) and windows[b + 1] or nil
        if w then
            local inp, tgt, n = w[1], w[2], w[3] or #w[1]
            if n > T then n = T end
            for t = 0, n - 1 do
                local a, y = inp[t + 1], tgt[t + 1]
                if a < 1 or a > V or y < 1 or y > V then
                    error(sformat("gpu_transformer: token out of range at window %d, pos %d " ..
                          "(input %s, target %s, vocab %d)",
                          b + 1, t + 1, tostring(a), tostring(y), V), 3)
                end
                ib[base + t] = a - 1        -- lanternl is 1-indexed, CUDA is 0
                tb[base + t] = y - 1
            end
            for t = n, T - 1 do
                ib[base + t] = 0
                tb[base + t] = -1           -- ignored: loss 0, dlogits 0
            end
            valid = valid + n
        else
            for t = 0, T - 1 do
                ib[base + t] = 0
                tb[base + t] = -1
            end
        end
    end

    luaTL.check(C.luaTL_upload_i32(self.ids.data, ib, N), "ids.upload")
    luaTL.check(C.luaTL_upload_i32(self.tgt.data, tb, N), "tgt.upload")
    return valid
end

function Model:_read_loss()
    luaTL.check(C.luaTL_download_f32(self.lossacc.data, self.f1, 1), "loss")
    return self.f1[0]
end

function Model:_read_gradnorm()
    if (self.clip_norm or 0) <= 0 then return 0 end
    luaTL.check(C.luaTL_download_f32(self.nsq.data, self.f1, 1), "gnorm")
    local v = self.f1[0]
    if v ~= v or v < 0 then return v end
    return sqrt(v)
end

function Model:_count_correct()
    if not self.want_acc then return 0 end
    local N = self.N
    luaTL.check(C.luaTL_download_i32(self.pred.data, self.h_pred, N), "pred")
    local pd, td = self.h_pred, self.h_tgt
    local c = 0
    for r = 0, N - 1 do
        local t = td[r]
        if t >= 0 and pd[r] == t then c = c + 1 end
    end
    return c
end

-- =====================================================================
--  Public API
-- =====================================================================

--- One optimizer step over up to B windows.
--- Returns loss (mean over valid tokens), correct, tokens, grad_norm.
function Model:train_step(windows, nwin, lr, step)
    local valid = self:_upload(windows, nwin)
    self:_set_valid(valid)

    self.p_zero:run(true)

    if self.graph_ready and self._cap_scale == self._valid then
        self.p_step:replay()
    else
        self.p_step:run(true)
    end

    local loss = self:_read_loss()
    local correct = self:_count_correct()

    -- match the CPU path: never apply a non-finite update
    local finite = (loss == loss) and loss ~= huge and loss ~= -huge
    local gnorm = 0
    if finite then
        self:_set_hparams(lr, step)
        self.p_opt:run(true)
        gnorm = self:_read_gradnorm()
        self.steps = (self.steps or 0) + 1
    end

    return loss, correct, valid, gnorm, finite
end

--- Forward-only loss/accuracy (validation).
function Model:eval_step(windows, nwin)
    local valid = self:_upload(windows, nwin)
    self:_set_valid(valid)
    self.p_fwd:run(true)
    return self:_read_loss(), self:_count_correct(), valid
end

--- Duck-typed forward for LMTrain:generate().  Returns an object shaped
--- like a tensor2 node: out.data.rows / .cols / .data[i] (1-indexed).
--- Only the final row is materialized -- that is all next-token sampling
--- reads, and it keeps a 50k-vocab head from creating a 100M-entry table.
function Model:forward(ids)
    local T, V, N = self.T, self.vocab, self.N
    local n = #ids
    if n < 1 then error("gpu_transformer: forward() needs at least one token", 2) end
    if n > T then
        local w = {}
        for i = n - T + 1, n do w[#w + 1] = ids[i] end
        ids, n = w, T
    end

    local tgt = {}
    for i = 1, n do tgt[i] = 1 end            -- unused; loss is discarded
    local valid = self:_upload({ { ids, tgt, n } }, 1)
    self:_set_valid(valid)
    self.p_fwd:run(true)

    self._row = self._row or ffi.new("float[?]", V)
    local rowptr = poff(self.logits, (n - 1) * V)
    luaTL.check(C.luaTL_download_f32(rowptr, self._row, V), "logits.row")

    local buf  = self._row
    local base = (n - 1) * V
    local proxy = setmetatable({}, { __index = function(_, i)
        local j = i - base
        if j < 1 or j > V then
            error("gpu_transformer: forward() materializes only the final " ..
                  "row of the logits (index " .. tostring(i) .. " is outside it)", 2)
        end
        return buf[j - 1]
    end })

    return { data = { rows = n, cols = V, data = proxy } }
end

--- Capture fwd+bwd as a CUDA graph.  Must be called after at least one
--- real step (so every lazy scratch allocation is already warm) and only
--- pays off when the valid-token count is stable, which is the common case.
function Model:capture()
    if self.graph_ready then return true end
    local ok, err = pcall(function()
        self.p_step:run(true)
        self.p_step:capture()
    end)
    if not ok then
        self.graph_error = tostring(err)
        return false, self.graph_error
    end
    self.graph_ready = true
    self._cap_scale  = self._valid
    return true
end

function Model:maybe_capture()
    if self.use_graph and not self.graph_ready and (self.steps or 0) >= 2 then
        self:capture()
    end
end

function Model:snapshot()
    local s = {}
    for i = 1, #self.params do s[i] = self.params[i].w:toflat() end
    return s
end

function Model:restore(snap)
    if not snap then return false end
    for i = 1, #self.params do
        if snap[i] then self.params[i].w:upload(snap[i]) end
    end
    return true
end

function Model:parameters() return self.params end

function Model:nparams()
    local n = 0
    for i = 1, #self.params do n = n + self.params[i].nelem end
    return n
end

--- FLOPs of one fwd+bwd step, counted GEMM by GEMM against what is
--- actually queued above.  Every forward GEMM costs 2*M*N*K; each has
--- exactly two backward counterparts of the same cost, hence the 3x.
function Model:flops_per_step()
    local B, T, D, H, hd = self.B, self.T, self.dim, self.heads, self.hd
    local F, V, L, N     = self.ffn, self.vocab, self.layers, self.N
    local f = 0
    for _ = 1, L do
        f = f + 2 * N * (3 * D) * D          -- qkv
        f = f + 2 * B * H * (2 * T * T * hd) -- QK^T and PV
        f = f + 2 * N * D * D                -- out proj
        f = f + 2 * N * (2 * F) * D          -- w13
        f = f + 2 * N * D * F                -- w2
    end
    f = f + 2 * N * V * D                    -- head
    return 3 * f, f
end

function Model:vram()
    local free, total = luaTL.mem_info()
    local st = luaTL.pool_stats()
    return { free = free, total = total, pool_in_use = st.in_use,
             pool_reserved = st.reserved, pool_peak = st.peak }
end

function Model:last_step_ms()
    return self.p_step.h.last_ms
end

function Model:free()
    if self.p_step then self.p_step:free_graph() end
    self.p_zero, self.p_fwd, self.p_step, self.p_opt = nil, nil, nil, nil
    for i = 1, #self._keep do
        local h = self._keep[i]
        if type(h) == "table" and h.free then h:free() end
    end
    for i = 1, #self.params do
        local pp = self.params[i]
        pp.w:free(); pp.g:free(); pp.m:free(); pp.v:free()
    end
    self._keep, self.params = {}, {}
end

return GT
