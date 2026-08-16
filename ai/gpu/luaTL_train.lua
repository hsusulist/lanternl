--     -- training extension for luaTL.lua --
--  Requires the library built from luaTL_train.cu.

local ffi   = require("ffi")
local luaTL = require("luaTL")
local C     = luaTL.C

ffi.cdef[[
typedef struct {
    int32_t  op, dtype, act, pre;
    int32_t  i0, i1, i2, i3, i4, i5, i6, i7;
    float    f0, f1, f2, f3, f4, f5, f6, f7;
    int64_t  s0, s1, s2, s3, s4, s5, s6, s7, s8;
    uint64_t n;
    void    *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7;
} luaTL_op_t;

typedef struct {
    luaTL_op_t* ops;
    int32_t     count, capacity, flags, status;
    void       *stream, *ev_start, *ev_stop, *graph, *graph_exec;
    double      last_ms;
    uint64_t    total_runs, total_ops;
    int32_t     owns_stream, timing, captured, capturing;
} luaTL_prog_t;

const char* luaTL_train_version(void);
uint64_t    luaTL_epoch(void);
void        luaTL_finalize(void);

int luaTL_gemm_ex(const void* A, const void* B, void* Cc,
                  const void* bias, const void* gamma, const float* rowscale,
                  int M, int N, int K, int dtype,
                  long long a_rs, long long a_cs,
                  long long b_rs, long long b_cs,
                  long long c_rs, long long c_cs,
                  long long a_bs, long long b_bs, long long c_bs,
                  int batch, float alpha, float beta, float eps,
                  int pre_op, int post_op);

int luaTL_rmsnorm_bwd(const void* x, const void* gamma, const void* dy,
                      void* dx, float* dgamma, int rows, int cols,
                      float eps, int dtype);
int luaTL_softmax_ex(const void* x, const void* bias, void* out,
                     int rows, int cols, float scale,
                     int causal, int group, int mask_offset, int dtype);
int luaTL_softmax_bwd(const void* y, const void* dy, void* dx,
                      int rows, int cols, float scale,
                      int causal, int group, int mask_offset, int dtype);
int luaTL_swiglu_fwd(const void* a, const void* b, void* out,
                     int rows, int cols, int lda, int ldb, int ldo, int dtype);
int luaTL_swiglu_bwd(const void* a, const void* b, const void* dout,
                     void* da, void* db, int rows, int cols,
                     int lda, int ldb, int ldo, int dtype);
int luaTL_rope(void* x, const void* pos_ids, int rows, int heads, int head_dim,
               int row_stride, int pos_offset, float theta,
               int interleaved, int inverse, int dtype);
int luaTL_embed_fwd(const void* ids, const void* table, void* out,
                    int rows, int dim, int vocab, int dtype);
int luaTL_embed_bwd(const void* ids, const void* dout, float* dtable,
                    int rows, int dim, int vocab, float scale, int dtype);
int luaTL_cross_entropy(const void* logits, const void* targets,
                        float* losses, void* dlogits,
                        int rows, int vocab, int ignore_index,
                        float grad_scale, int dtype);
int luaTL_reduce_cols(const void* x, float* out, int rows, int cols,
                      long long row_stride, float alpha, float beta, int dtype);
int luaTL_l2_accum(const void* x, uint64_t n, float* out, int dtype);
int luaTL_clip_by_nsq(void* x, uint64_t n, const float* nsq,
                      float max_norm, int dtype);
int luaTL_sum_f32(const float* x, uint64_t n, float* out, float scale);
int luaTL_argmax_rows(const void* x, void* out_i32, int rows, int cols, int dtype);
int luaTL_gather_rows(const void* src, const void* idx, void* dst,
                      int n, int cols, int src_rows, int dtype);
int luaTL_row_rms(const void* x, float* out, int rows, int cols,
                  long long rs, long long cs, float eps, int dtype);
int luaTL_upload_i32(void* dev, const int32_t* host, uint64_t n);
int luaTL_download_i32(const void* dev, int32_t* host, uint64_t n);
int luaTL_download_f32(const void* dev, float* host, uint64_t n);

luaTL_prog_t* luaTL_prog_create(int capacity, int timing);
void          luaTL_prog_destroy(luaTL_prog_t* P);
void          luaTL_prog_reset(luaTL_prog_t* P);
int           luaTL_prog_run(luaTL_prog_t* P);
int           luaTL_prog_capture(luaTL_prog_t* P);
int           luaTL_prog_replay(luaTL_prog_t* P);
int           luaTL_prog_wait(luaTL_prog_t* P);
int           luaTL_prog_push(luaTL_prog_t* P, const luaTL_op_t* op);
void          luaTL_prog_graph_free(luaTL_prog_t* P);
]]

local T      = luaTL.TensorClass
local ck     = luaTL.check
local F32    = luaTL.F32
local i32_ct = ffi.typeof("int32_t[?]")
local f32_ct = ffi.typeof("float[?]")

luaTL.POP = {
    NOP=0, GEMM=1, EW=2, RMSNORM=3, RMSNORM_BWD=4, SOFTMAX=5, SOFTMAX_BWD=6,
    SWIGLU=7, SWIGLU_BWD=8, ROPE=9, EMBED=10, EMBED_BWD=11, CE=12,
    REDUCE_COLS=13, L2=14, CLIP=15, ADAMW=16, ZERO=17, COPY=18, CAST=19,
    TRANSPOSE=20, BCAST_ADD=21, SYNC=22, ARGMAX=23, SUM=24, GATHER=25,
    ROWRMS=26,
}

luaTL.train_version = ffi.string(C.luaTL_train_version())

function luaTL.finalize() C.luaTL_finalize() end

--  Strided views: describe transposes / head slices WITHOUT copying.
--  A "stride view" is a plain Lua table, not a tensor -- it just carries
--  a pointer plus the six strides that gemm_ex needs.
local function sv(ptr, rows, cols, rs, cs, dtype, bs)
    return { data = ptr, rows = rows, cols = cols,
             rs = rs, cs = cs, bs = bs or 0, dtype = dtype or F32 }
end
luaTL.sv = sv

--- Logical transpose. Zero cost: swaps the two strides.
function T:t()
    return sv(self.data, self.cols, self.rows, 1, self.cols, self.dtype, 0)
end

--- Plain row-major view of the whole tensor.
function T:as_sv()
    return sv(self.data, self.rows, self.cols, self.cols, 1, self.dtype, 0)
end

--- Column slice [., col0 .. col0+w-1] keeping the parent leading dim.
--- This is how you read Q, K, V out of one fused [T, 3D] projection.
function T:cols_view(col0, w)
    local esz = luaTL.dtype_size[self.dtype]
    local p   = ffi.cast("char*", self.data) + col0 * esz
    return sv(p, self.rows, w, self.cols, 1, self.dtype, 0)
end

--- Head h of a [rows, heads*hd] activation, as a batched view.
--- batch stride = hd, so `batch = heads` sweeps every head in one launch.
function T:head_view(hd, heads)
    return sv(self.data, self.rows, hd, heads * hd, 1, self.dtype, hd)
end

local function coerce(x)
    -- accepts a Tensor or a stride view
    if type(x) == "table" then return x end
    return { data = x.data, rows = x.rows, cols = x.cols,
             rs = x.cols, cs = 1, bs = 0, dtype = x.dtype }
end
luaTL.coerce = coerce

--  Immediate-mode strided GEMM
--    Cc = act(alpha * (pre(A) @ B) + bias) + beta*Cc
--  A, B, Cc may each be a Tensor or a stride view.
function luaTL.gemm_ex(A, B, Cc, o)
    o = o or {}
    local a, b, c = coerce(A), coerce(B), coerce(Cc)
    local M, K, N = a.rows, a.cols, b.cols
    if b.rows ~= K then
        error(string.format("gemm_ex: inner dim mismatch %dx%d @ %dx%d",
                            a.rows, a.cols, b.rows, b.cols), 2)
    end
    ck(C.luaTL_gemm_ex(a.data, b.data, c.data,
        o.bias and o.bias.data or nil,
        o.gamma and o.gamma.data or nil,
        o.rowscale and ffi.cast("float*", o.rowscale.data) or nil,
        M, N, K, c.dtype,
        a.rs, a.cs, b.rs, b.cs, c.rs, c.cs,
        a.bs, b.bs, c.bs, o.batch or 1,
        o.alpha or 1.0, o.beta or 0.0, o.eps or 1e-5,
        o.pre or 0, o.act or 0), "gemm_ex")
    return Cc
end

--- dX = dY @ W^T   (W stored [K,N]; we need [N,K] -> logical transpose)
function luaTL.matmul_dx(dY, W, dX, beta)
    return luaTL.gemm_ex(dY, W:t(), dX, { beta = beta or 0.0 })
end

--- dW = X^T @ dY   (accumulating by default: beta = 1)
function luaTL.matmul_dw(X, dY, dW, beta)
    return luaTL.gemm_ex(X:t(), dY, dW, { beta = beta == nil and 1.0 or beta })
end

--  Immediate-mode wrappers for the new kernels
function T:rmsnorm_bwd(gamma, dy, dx, dgamma, eps)
    ck(C.luaTL_rmsnorm_bwd(self.data, gamma and gamma.data or nil, dy.data,
        dx.data, dgamma and ffi.cast("float*", dgamma.data) or nil,
        self.rows, self.cols, eps or 1e-5, self.dtype), "rmsnorm_bwd")
    return dx
end

function T:softmax_ex(out, o)
    o = o or {}
    ck(C.luaTL_softmax_ex(self.data, o.bias and o.bias.data or nil,
        (out or self).data, self.rows, self.cols, o.scale or 1.0,
        o.causal and 1 or 0, o.group or 0, o.mask_offset or 0, self.dtype),
       "softmax_ex")
    return out or self
end

function T:softmax_bwd(dy, dx, o)
    o = o or {}
    ck(C.luaTL_softmax_bwd(self.data, dy.data, dx.data, self.rows, self.cols,
        o.scale or 1.0, o.causal and 1 or 0, o.group or 0,
        o.mask_offset or 0, self.dtype), "softmax_bwd")
    return dx
end

function luaTL.swiglu(a, b, out, o)
    o = o or {}
    ck(C.luaTL_swiglu_fwd(a.data, b.data, out.data, out.rows, out.cols,
        o.lda or a.cols, o.ldb or b.cols, o.ldo or out.cols, out.dtype),
       "swiglu")
    return out
end

function luaTL.swiglu_bwd(a, b, dout, da, db, o)
    o = o or {}
    ck(C.luaTL_swiglu_bwd(a.data, b.data, dout.data,
        da and da.data or nil, db and db.data or nil,
        dout.rows, dout.cols, o.lda or a.cols, o.ldb or b.cols,
        o.ldo or dout.cols, dout.dtype), "swiglu_bwd")
end

function T:rope(o)
    o = o or {}
    local heads = o.heads or 1
    local hd    = o.head_dim or (self.cols / heads)
    ck(C.luaTL_rope(self.data, o.pos_ids and o.pos_ids.data or nil,
        self.rows, heads, hd, o.row_stride or self.cols,
        o.pos_offset or 0, o.theta or 10000.0,
        o.interleaved and 1 or 0, o.inverse and 1 or 0, self.dtype), "rope")
    return self
end

function luaTL.embed(ids, tbl, out)
    ck(C.luaTL_embed_fwd(ids.data, tbl.data, out.data,
        out.rows, out.cols, tbl.rows, out.dtype), "embed")
    return out
end

function luaTL.embed_bwd(ids, dout, dtable, scale)
    ck(C.luaTL_embed_bwd(ids.data, dout.data,
        ffi.cast("float*", dtable.data), dout.rows, dout.cols,
        dtable.rows, scale or 1.0, dout.dtype), "embed_bwd")
end

--- Fused softmax + cross-entropy. `losses` is a [rows,1] f32 tensor,
--- `dlogits` may alias `logits` (written in place). Returns mean loss.
function luaTL.cross_entropy(logits, targets, losses, dlogits, o)
    o = o or {}
    ck(C.luaTL_cross_entropy(logits.data, targets.data,
        ffi.cast("float*", losses.data), dlogits and dlogits.data or nil,
        logits.rows, logits.cols, o.ignore_index or -1,
        o.grad_scale or (1.0 / logits.rows), logits.dtype), "cross_entropy")
    local buf = f32_ct(logits.rows)
    ck(C.luaTL_download_f32(losses.data, buf, logits.rows), "ce.download")
    local s = 0.0
    for i = 0, logits.rows - 1 do s = s + buf[i] end
    return s / logits.rows
end

function T:reduce_cols(out, alpha, beta, row_stride)
    ck(C.luaTL_reduce_cols(self.data, ffi.cast("float*", out.data),
        self.rows, self.cols, row_stride or self.cols,
        alpha or 1.0, beta or 0.0, self.dtype), "reduce_cols")
    return out
end

function T:l2_accum(acc)
    ck(C.luaTL_l2_accum(self.data, self.nelem,
        ffi.cast("float*", acc.data), self.dtype), "l2_accum")
    return acc
end

function T:clip_by(nsq, max_norm)
    ck(C.luaTL_clip_by_nsq(self.data, self.nelem,
        ffi.cast("const float*", nsq.data), max_norm, self.dtype), "clip")
    return self
end

function T:argmax(out_i32)
    ck(C.luaTL_argmax_rows(self.data, out_i32.data,
        self.rows, self.cols, self.dtype), "argmax")
    return out_i32
end

function T:row_rms(out, eps, rs, cs)
    ck(C.luaTL_row_rms(self.data, ffi.cast("float*", out.data),
        self.rows, self.cols, rs or self.cols, cs or 1,
        eps or 1e-5, self.dtype), "row_rms")
    return out
end

-- ---- int32 index tensors (tokens / targets) -------------------------
--- Backed by an f32 tensor's storage; 4 bytes per element either way.
function luaTL.ids(n)
    local t = luaTL.zeros(n, 1, F32)
    t.flags = 1                      -- mark as index buffer
    return t
end

function luaTL.upload_ids(t, list, one_indexed)
    local n   = #list
    local buf = i32_ct(n)
    local off = one_indexed and 1 or 0
    for i = 1, n do buf[i - 1] = list[i] - off end
    ck(C.luaTL_upload_i32(t.data, buf, n), "upload_ids")
    return t
end

function luaTL.download_ids(t, n, one_indexed)
    n = n or tonumber(t.nelem)
    local buf = i32_ct(n)
    ck(C.luaTL_download_i32(t.data, buf, n), "download_ids")
    local out, off = {}, one_indexed and 1 or 0
    for i = 1, n do out[i] = buf[i - 1] + off end
    return out
end

function luaTL.scalar_value(t)
    local buf = f32_ct(1)
    ck(C.luaTL_download_f32(t.data, buf, 1), "scalar_value")
    return buf[0]
end

--  Program : the wide command buffer + CUDA Graph replay
local Prog = {}
Prog.__index = Prog
luaTL.ProgramClass = Prog

function luaTL.Program(capacity, timing)
    local h = C.luaTL_prog_create(capacity or 4096, timing and 1 or 0)
    if h == nil then error("luaTL.Program: " .. luaTL.last_error(), 2) end
    return setmetatable({ h = ffi.gc(h, C.luaTL_prog_destroy) }, Prog)
end

function Prog:reset()
    self.h.count  = 0
    self.h.status = 0
    return self
end

function Prog:slot()
    local h = self.h
    local i = h.count
    if i >= h.capacity then
        error("luaTL.Program full (" .. tonumber(h.capacity) .. ")", 3)
    end
    h.count = i + 1
    local c = h.ops[i]
    c.op, c.dtype, c.act, c.pre = 0, 0, 0, 0
    c.i0, c.i1, c.i2, c.i3 = 0, 0, 0, 0
    c.i4, c.i5, c.i6, c.i7 = 0, 0, 0, 0
    c.f0, c.f1, c.f2, c.f3 = 0, 0, 0, 0
    c.f4, c.f5, c.f6, c.f7 = 0, 0, 0, 0
    c.s0, c.s1, c.s2, c.s3, c.s4 = 0, 0, 0, 0, 0
    c.s5, c.s6, c.s7, c.s8 = 0, 0, 0, 0
    c.n = 0
    c.p0, c.p1, c.p2, c.p3 = nil, nil, nil, nil
    c.p4, c.p5, c.p6, c.p7 = nil, nil, nil, nil
    return c
end

function Prog:gemm(A, B, Cc, o)
    o = o or {}
    local a, b, cc = coerce(A), coerce(B), coerce(Cc)
    local c = self:slot()
    c.op    = 1
    c.dtype = cc.dtype
    c.act   = o.act or 0
    c.pre   = o.pre or 0
    c.i0, c.i1, c.i2 = a.rows, b.cols, a.cols
    c.i3 = o.batch or 1
    c.f0 = o.alpha or 1.0
    c.f1 = o.beta  or 0.0
    c.f2 = o.eps   or 1e-5
    c.s0, c.s1 = a.rs, a.cs
    c.s2, c.s3 = b.rs, b.cs
    c.s4, c.s5 = cc.rs, cc.cs
    c.s6, c.s7, c.s8 = a.bs, b.bs, cc.bs
    c.p0, c.p1, c.p2 = a.data, b.data, cc.data
    c.p3 = o.bias  and o.bias.data  or nil
    c.p4 = o.gamma and o.gamma.data or nil
    c.p5 = o.rowscale and o.rowscale.data or nil
    return self
end

function Prog:matmul(A, B, Cc, o) return self:gemm(A, B, Cc, o) end
function Prog:dx(dY, W, dX, beta)
    return self:gemm(dY, W:t(), dX, { beta = beta or 0.0 })
end
function Prog:dw(X, dY, dW, beta)
    return self:gemm(X:t(), dY, dW, { beta = beta == nil and 1.0 or beta })
end

function Prog:ew(op, A, B, out, alpha, beta)
    local c = self:slot()
    c.op, c.dtype, c.i0 = 2, out.dtype, op
    c.f0, c.f1 = alpha or 1.0, beta or 0.0
    c.n  = out.nelem
    c.p0 = A and A.data or nil
    c.p1 = B and B.data or nil
    c.p2 = out.data
    return self
end
function Prog:add(A, B, out) return self:ew(luaTL.EW.ADD, A, B, out, 1, 1) end
function Prog:axpy(dst, B, a) return self:ew(luaTL.EW.AXPY, dst, B, dst, a or 1, 0) end
function Prog:scale(dst, a)   return self:ew(luaTL.EW.SCALE, dst, nil, dst, a, 0) end
function Prog:dsilu(x, dy, out) return self:ew(luaTL.EW.DSILU, x, dy, out, 1, 0) end
function Prog:dgelu(x, dy, out) return self:ew(luaTL.EW.DGELU, x, dy, out, 1, 0) end

function Prog:rmsnorm(x, w, out, eps)
    local c = self:slot()
    c.op, c.dtype = 3, x.dtype
    c.i0, c.i1, c.f0 = x.rows, x.cols, eps or 1e-5
    c.p0, c.p1, c.p2 = x.data, w and w.data or nil, out.data
    return self
end

function Prog:rmsnorm_bwd(x, gamma, dy, dx, dgamma, eps)
    local c = self:slot()
    c.op, c.dtype = 4, x.dtype
    c.i0, c.i1, c.f0 = x.rows, x.cols, eps or 1e-5
    c.p0, c.p1, c.p2 = x.data, gamma and gamma.data or nil, dy.data
    c.p3 = dx.data
    c.p4 = dgamma and dgamma.data or nil
    return self
end

function Prog:softmax(x, out, o)
    o = o or {}
    local c = self:slot()
    c.op, c.dtype = 5, x.dtype
    c.i0, c.i1 = x.rows, x.cols
    c.i2 = o.causal and 1 or 0
    c.i3 = o.group or 0
    c.i4 = o.mask_offset or 0
    c.f0 = o.scale or 1.0
    c.p0, c.p1, c.p2 = x.data, o.bias and o.bias.data or nil, out.data
    return self
end

function Prog:softmax_bwd(y, dy, dx, o)
    o = o or {}
    local c = self:slot()
    c.op, c.dtype = 6, y.dtype
    c.i0, c.i1 = y.rows, y.cols
    c.i2 = o.causal and 1 or 0
    c.i3 = o.group or 0
    c.i4 = o.mask_offset or 0
    c.f0 = o.scale or 1.0
    c.p0, c.p1, c.p2 = y.data, dy.data, dx.data
    return self
end

function Prog:swiglu(a, b, out, o)
    o = o or {}
    local c = self:slot()
    c.op, c.dtype = 7, out.dtype
    c.i0, c.i1 = out.rows, out.cols
    c.i2, c.i3, c.i4 = o.lda or a.cols, o.ldb or b.cols, o.ldo or out.cols
    c.p0, c.p1, c.p2 = a.data, b.data, out.data
    return self
end

function Prog:swiglu_bwd(a, b, dout, da, db, o)
    o = o or {}
    local c = self:slot()
    c.op, c.dtype = 8, dout.dtype
    c.i0, c.i1 = dout.rows, dout.cols
    c.i2, c.i3, c.i4 = o.lda or a.cols, o.ldb or b.cols, o.ldo or dout.cols
    c.p0, c.p1, c.p2 = a.data, b.data, dout.data
    c.p3 = da and da.data or nil
    c.p4 = db and db.data or nil
    return self
end

function Prog:rope(x, o)
    o = o or {}
    local heads = o.heads or 1
    local c = self:slot()
    c.op, c.dtype = 9, x.dtype
    c.i0, c.i1 = x.rows, heads
    c.i2 = o.head_dim or (x.cols / heads)
    c.i3 = o.row_stride or x.cols
    c.i4 = o.pos_offset or 0
    c.i5 = o.interleaved and 1 or 0
    c.i6 = o.inverse and 1 or 0
    c.f0 = o.theta or 10000.0
    c.p0 = x.data
    c.p1 = o.pos_ids and o.pos_ids.data or nil
    return self
end

function Prog:embed(ids, tbl, out)
    local c = self:slot()
    c.op, c.dtype = 10, out.dtype
    c.i0, c.i1, c.i2 = out.rows, out.cols, tbl.rows
    c.p0, c.p1, c.p2 = ids.data, tbl.data, out.data
    return self
end

function Prog:embed_bwd(ids, dout, dtable, scale)
    local c = self:slot()
    c.op, c.dtype = 11, dout.dtype
    c.i0, c.i1, c.i2 = dout.rows, dout.cols, dtable.rows
    c.f0 = scale or 1.0
    c.p0, c.p1, c.p2 = ids.data, dout.data, dtable.data
    return self
end

function Prog:cross_entropy(logits, targets, losses, dlogits, o)
    o = o or {}
    local c = self:slot()
    c.op, c.dtype = 12, logits.dtype
    c.i0, c.i1 = logits.rows, logits.cols
    c.i2 = o.ignore_index or -1
    c.f0 = o.grad_scale or (1.0 / logits.rows)
    c.p0, c.p1, c.p2 = logits.data, targets.data, losses.data
    c.p3 = dlogits and dlogits.data or nil
    return self
end

function Prog:reduce_cols(x, out, alpha, beta, rs)
    local c = self:slot()
    c.op, c.dtype = 13, x.dtype
    c.i0, c.i1 = x.rows, x.cols
    c.s0 = rs or x.cols
    c.f0, c.f1 = alpha or 1.0, beta or 0.0
    c.p0, c.p2 = x.data, out.data
    return self
end

function Prog:l2(x, acc)
    local c = self:slot()
    c.op, c.dtype, c.n = 14, x.dtype, x.nelem
    c.p0, c.p2 = x.data, acc.data
    return self
end

function Prog:clip(x, nsq, max_norm)
    local c = self:slot()
    c.op, c.dtype, c.n = 15, x.dtype, x.nelem
    c.f0 = max_norm
    c.p1, c.p2 = nsq.data, x.data
    return self
end

function Prog:adamw(w, g, m, v, w_lp, o)
    o = o or {}
    local c = self:slot()
    c.op, c.dtype = 16, g.dtype
    c.i0 = o.step or 1
    c.f0, c.f1, c.f2 = o.lr or 1e-3, o.beta1 or 0.9, o.beta2 or 0.999
    c.f3, c.f4, c.f5 = o.eps or 1e-8, o.wd or 0.0, o.clip or 0.0
    c.n = w.nelem
    c.p0, c.p1, c.p2, c.p3 = w.data, g.data, m.data, v.data
    c.p4 = w_lp and w_lp.data or nil
    return self
end

function Prog:zero(t)
    local c = self:slot(); c.op, c.n, c.p2 = 17, t.nbytes, t.data; return self
end
function Prog:copy(src, dst)
    local c = self:slot(); c.op, c.n = 18, src.nbytes
    c.p0, c.p2 = src.data, dst.data; return self
end
function Prog:cast(src, dst)
    local c = self:slot(); c.op = 19
    c.i0, c.i1, c.n = src.dtype, dst.dtype, src.nelem
    c.p0, c.p2 = src.data, dst.data; return self
end
function Prog:transpose(x, out)
    local c = self:slot(); c.op, c.dtype = 20, x.dtype
    c.i0, c.i1 = x.rows, x.cols
    c.p0, c.p2 = x.data, out.data; return self
end
function Prog:bcast_add(x, vec, out, alpha)
    local c = self:slot(); c.op, c.dtype = 21, x.dtype
    c.i0, c.i1, c.f0 = x.rows, x.cols, alpha or 1.0
    c.p0, c.p1, c.p2 = x.data, vec.data, out.data; return self
end
function Prog:barrier() self:slot().op = 22 return self end
function Prog:argmax(x, out)
    local c = self:slot(); c.op, c.dtype = 23, x.dtype
    c.i0, c.i1 = x.rows, x.cols
    c.p0, c.p2 = x.data, out.data; return self
end
function Prog:sum(x, acc, scale)
    local c = self:slot(); c.op, c.n, c.f0 = 24, x.nelem, scale or 1.0
    c.p0, c.p2 = x.data, acc.data; return self
end
function Prog:gather(src, idx, dst)
    local c = self:slot(); c.op, c.dtype = 25, dst.dtype
    c.i0, c.i1, c.i2 = dst.rows, dst.cols, src.rows
    c.p0, c.p1, c.p2 = src.data, idx.data, dst.data; return self
end
function Prog:row_rms(x, out, eps, rs, cs)
    local c = self:slot(); c.op, c.dtype = 26, x.dtype
    c.i0, c.i1 = x.rows, x.cols
    c.s0, c.s1, c.f0 = rs or x.cols, cs or 1, eps or 1e-5
    c.p0, c.p2 = x.data, out.data; return self
end

function Prog:async(on) self.h.flags = on and 0 or 1 return self end

function Prog:run(keep)
    local rc = C.luaTL_prog_run(self.h)
    if rc ~= 0 then error("Program:run: " .. luaTL.last_error(), 2) end
    if not keep then self.h.count = 0 end
    return self
end

--- Capture the CURRENT op list as a CUDA graph. Run once first.
function Prog:capture()
    ck(C.luaTL_prog_capture(self.h), "prog.capture")
    return self
end

function Prog:replay()
    local rc = C.luaTL_prog_replay(self.h)
    if rc ~= 0 then error("Program:replay: " .. luaTL.last_error(), 2) end
    return self
end

function Prog:wait() ck(C.luaTL_prog_wait(self.h), "prog.wait") return self end
function Prog:elapsed_ms() return self.h.last_ms end
function Prog:size() return self.h.count end
function Prog:is_captured() return self.h.captured ~= 0 end
function Prog:free_graph() C.luaTL_prog_graph_free(self.h) return self end

return luaTL