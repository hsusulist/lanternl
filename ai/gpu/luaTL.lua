-- =====================================================================
--  luaTL.lua  —  LuaJIT FFI binding + zero-allocation tensor engine
--                for luaTL_core 2.0
--
--  Design notes
--  ------------
--   * A tensor is a `luaTL_tensor_t` cdata struct with an ffi.metatype,
--     NOT a Lua table.  There is no hash lookup on the hot path and the
--     GC sees exactly one object per tensor.
--   * Every C entry point returns an int status; `ck()` turns a non-zero
--     status into a Lua error.  Hot paths use the `_raw` variants that
--     skip the check entirely.
--   * The pipeline lets Lua describe a whole forward step with plain
--     struct stores and execute it with ONE FFI call.
-- =====================================================================

local ffi  = require("ffi")
local bit  = require("bit")

ffi.cdef[[
/* ---- POD structs (byte identical to luaTL_core.cu) ---------------- */
typedef struct {
    int32_t  device, major, minor, mp_count, warp_size;
    int32_t  max_threads_per_block, max_threads_per_mp, max_blocks_per_mp;
    int32_t  regs_per_block, regs_per_mp, clock_khz, memory_clock_khz;
    int32_t  memory_bus_width, l2_cache_size, concurrent_kernels;
    int32_t  async_engines, unified_addressing, cooperative_launch;
    int32_t  has_tensor_cores, has_fp16, has_bf16, ecc_enabled;
    int32_t  integrated, cores_per_mp;
    uint64_t shared_mem_per_block, shared_mem_per_mp;
    uint64_t total_global_mem, free_global_mem;
    double   peak_mem_bandwidth_gbs, peak_fp32_gflops, peak_fp16_gflops;
    char     name[256];
} luaTL_devinfo_t;

typedef struct {
    int32_t tm, tn, tk, block_x, block_y, grid_x, grid_y;
    int32_t shmem_bytes, kernel_id, unroll, use_tensor_core, from_cache;
    double  measured_ms;
} luaTL_plan_t;

typedef struct {
    uint64_t bytes_reserved, bytes_in_use, peak_in_use;
    uint64_t block_count, free_block_count;
    uint64_t n_alloc, n_free, n_cuda_malloc, n_cuda_free;
    uint64_t n_cache_hit, n_cache_miss, n_trim;
} luaTL_pool_stats_t;

typedef struct luaTL_pool_s luaTL_pool_t;

typedef struct {
    void*    data;
    uint64_t nelem, nbytes;
    int32_t  rows, cols, dtype, owns;
    void*    pool;
    int32_t  device, flags;
} luaTL_tensor_t;

typedef struct {
    char*    base;
    uint64_t capacity, offset, peak, alignment;
    void*    pool;
} luaTL_arena_t;

typedef struct {
    int32_t  op, dtype, act, flags;
    int32_t  i0, i1, i2, i3;
    float    f0, f1, f2, f3, f4, f5;
    uint64_t n;
    void    *p0, *p1, *p2, *p3, *p4, *p5;
} luaTL_cmd_t;

typedef struct {
    luaTL_cmd_t* cmds;
    int32_t      count, capacity, flags, status;
    void        *stream, *ev_start, *ev_stop;
    double       last_ms;
    uint64_t     total_runs, total_ops;
    int32_t      owns_stream, timing;
} luaTL_pipeline_t;

/* ---- runtime ------------------------------------------------------ */
const char*  luaTL_version(void);
void         luaTL_set_verbose(int on);
int          luaTL_init(int device);
void         luaTL_shutdown(void);
const char*  luaTL_device_name(void);
const char*  luaTL_get_last_error(void);
int          luaTL_has_error(void);
void         luaTL_clear_error(void);
int          luaTL_sync(void);
int          luaTL_get_devinfo(int device, luaTL_devinfo_t* out);
int          luaTL_mem_info(uint64_t* freeb, uint64_t* totalb);

/* ---- streams / pinned host memory --------------------------------- */
void*        luaTL_stream_create(int nonblocking);
void         luaTL_stream_destroy(void* s);
int          luaTL_stream_sync(void* s);
void*        luaTL_host_alloc_pinned(uint64_t bytes);
void         luaTL_host_free_pinned(void* p);

/* ---- pool --------------------------------------------------------- */
luaTL_pool_t* luaTL_pool_create(int device);
void          luaTL_pool_destroy(luaTL_pool_t* P);
luaTL_pool_t* luaTL_default_pool(void);
void*         luaTL_tensor_alloc(luaTL_pool_t* P, uint64_t bytes);
void          luaTL_tensor_free(luaTL_pool_t* P, void* ptr);
void          luaTL_pool_trim(luaTL_pool_t* P);
int           luaTL_pool_get_stats(luaTL_pool_t* P, luaTL_pool_stats_t* out);
int           luaTL_pool_owns(luaTL_pool_t* P, void* ptr);

/* ---- arena -------------------------------------------------------- */
luaTL_arena_t* luaTL_arena_create(luaTL_pool_t* P, uint64_t bytes);
void           luaTL_arena_destroy(luaTL_arena_t* A);
void*          luaTL_arena_alloc(luaTL_arena_t* A, uint64_t bytes);
void           luaTL_arena_reset(luaTL_arena_t* A);

/* ---- tensors ------------------------------------------------------ */
int  luaTL_tensor_init(luaTL_tensor_t* t, int rows, int cols, int dtype, luaTL_pool_t* P);
int  luaTL_tensor_wrap(luaTL_tensor_t* t, void* data, int rows, int cols, int dtype);
void luaTL_tensor_release(luaTL_tensor_t* t);
int  luaTL_tensor_view(luaTL_tensor_t* dst, const luaTL_tensor_t* src, int row_off, int rows, int cols);
int  luaTL_tensor_zero(luaTL_tensor_t* t);
int  luaTL_tensor_upload(luaTL_tensor_t* t, const float* host, uint64_t nelem);
int  luaTL_tensor_download(const luaTL_tensor_t* t, float* host, uint64_t nelem);
int  luaTL_tensor_copy(const luaTL_tensor_t* src, luaTL_tensor_t* dst);
int  luaTL_tensor_cast(const luaTL_tensor_t* src, luaTL_tensor_t* dst);

/* ---- legacy raw pointer API --------------------------------------- */
float* luaTL_gpu_malloc(uint64_t nelem);
void   luaTL_gpu_free(float* p);
int    luaTL_to_gpu(const float* host, float* dev, uint64_t nelem);
int    luaTL_to_cpu(const float* dev, float* host, uint64_t nelem);
int    luaTL_gpu_copy(const float* src, float* dst, uint64_t nelem);
int    luaTL_gpu_zero(float* p, uint64_t nelem);

/* ---- autotuner ---------------------------------------------------- */
int  luaTL_plan_gemm(int M, int N, int K, int dtype, int allow_tc, luaTL_plan_t* out);
void luaTL_set_tile_override(int kernel_id);
void luaTL_tune_reset(void);
int  luaTL_autotune_benchmark(int M, int N, int K, int dtype, int iters, luaTL_plan_t* out);

/* ---- math --------------------------------------------------------- */
int luaTL_gemm(const void* A, const void* B, void* C, const void* bias,
               const void* gamma, const float* rowscale,
               int M, int N, int K, int dtype,
               float alpha, float beta, float eps, int pre_op, int post_op);
int matmul_cuda(const float* A, const float* B, float* C, int M, int N, int K);
int fused_rmsnorm_matmul_cuda(const float* x, const float* gamma, const float* W,
                              float* out, const float* bias, int M, int N, int K, float eps);
int fused_gelu_matmul_cuda(const float* x, const float* W, float* out,
                           const float* bias, int M, int N, int K);
int fused_silu_matmul_cuda(const float* x, const float* W, float* out,
                           const float* bias, int M, int N, int K);
int fused_matmul_gelu_cuda(const float* A, const float* B, float* C,
                           const float* bias, int M, int N, int K);
int luaTL_elementwise(const void* A, const void* B, void* C, uint64_t n,
                      int dtype, int op, float alpha, float beta);
int add_cuda(const float* A, const float* B, float* C, uint64_t n);
int add_inplace_cuda(float* A, const float* B, uint64_t n, float alpha);
int luaTL_rmsnorm(const void* x, const void* w, void* out,
                  int rows, int cols, float eps, int dtype);
int rmsnorm_cuda(const float* x, const float* w, float* out, int rows, int cols, float eps);
int rmsnorm_inplace_cuda(float* x, const float* w, int rows, int cols, float eps);
int luaTL_softmax(const void* x, const void* bias, void* out, int rows, int cols,
                  float scale, int causal, int mask_offset, int dtype);
int softmax_cuda(const float* x, float* out, int rows, int cols);
int fused_add_bias_softmax_cuda(const float* x, const float* bias, float* out,
                                int rows, int cols, float scale, int causal);
int luaTL_bcast_add(const void* x, const void* v, void* out,
                    int rows, int cols, float alpha, int dtype);
int luaTL_transpose(const void* in, void* out, int rows, int cols, int dtype);
int luaTL_rms_rowscale(const void* x, float* scale, int rows, int cols, float eps, int dtype);
int luaTL_adamw(float* w, const void* g, float* m, float* v, void* w_lp,
                uint64_t n, int grad_dtype, float lr, float beta1, float beta2,
                float eps, float wd, int step, float clip);

/* ---- pipeline ------------------------------------------------------ */
luaTL_pipeline_t* luaTL_pipeline_create(int capacity, int timing);
void              luaTL_pipeline_destroy(luaTL_pipeline_t* P);
void              luaTL_pipeline_reset(luaTL_pipeline_t* P);
int               luaTL_pipeline_push(luaTL_pipeline_t* P, const luaTL_cmd_t* cmd);
int               luaTL_pipeline_run(luaTL_pipeline_t* P);
int               luaTL_pipeline_wait(luaTL_pipeline_t* P);
]]

-- ---------------------------------------------------------------------
--  Library loading
-- ---------------------------------------------------------------------
local CANDIDATES
if ffi.os == "Windows" then
    CANDIDATES = { "luaTL", "luaTL.dll", ".\\luaTL.dll" }
else
    CANDIDATES = { "luaTL", "./luaTL.so", "libluaTL.so", "./libluaTL.so" }
end

local C, load_err
for _, name in ipairs(CANDIDATES) do
    local ok, lib = pcall(ffi.load, name)
    if ok then C = lib break else load_err = lib end
end
if not C then
    error("luaTL: cannot load the native library (" .. tostring(load_err) ..
          "). Build it with nvcc first.", 2)
end

local luaTL = { _VERSION = "luaTL 2.0", C = C }

-- ---------------------------------------------------------------------
--  Constants
-- ---------------------------------------------------------------------
luaTL.F32, luaTL.F16, luaTL.BF16 = 0, 1, 2

luaTL.dtype = { f32 = 0, float = 0, f16 = 1, half = 1, bf16 = 2, bfloat16 = 2 }
luaTL.dtype_name = { [0] = "f32", [1] = "f16", [2] = "bf16" }
luaTL.dtype_size = { [0] = 4, [1] = 2, [2] = 2 }

luaTL.EW = {
    COPY = 0, ADD = 1, SUB = 2, MUL = 3, DIV = 4, SCALE = 5, AXPY = 6,
    FILL = 7, RELU = 8, GELU = 9, SILU = 10, TANH = 11, SIGMOID = 12,
    EXP = 13, SQRT = 14, RSQRT = 15, NEG = 16, RECIP = 17, ABS = 18,
    CLAMP = 19, DRELU = 20, DSILU = 21, DGELU = 22,
}

luaTL.ACT = { NONE = 0, RELU = 1, GELU = 2, SILU = 3, TANH = 4, SIGMOID = 5 }
luaTL.PRE = { NONE = 0, RMSNORM = 1, GELU = 2, SILU = 3, RELU = 4 }

luaTL.OP = {
    NOP = 0, GEMM = 1, EW = 2, RMSNORM = 3, SOFTMAX = 4, CAST = 5,
    TRANSPOSE = 6, BCAST_ADD = 7, ADAMW = 8, ZERO = 9, COPY = 10, SYNC = 11,
}

local STATUS_NAME = {
    [0] = "OK", [-1] = "NULL_POINTER", [-2] = "SHAPE", [-3] = "DTYPE",
    [-4] = "CUDA", [-5] = "OUT_OF_MEMORY", [-6] = "BAD_ARGUMENT",
    [-7] = "CAPACITY", [-8] = "UNSUPPORTED",
}

-- ---------------------------------------------------------------------
--  Cached ctypes and reusable scratch cdata (avoids per-call allocation)
-- ---------------------------------------------------------------------
local tensor_ct  = ffi.typeof("luaTL_tensor_t")
local devinfo_ct = ffi.typeof("luaTL_devinfo_t")
local plan_ct    = ffi.typeof("luaTL_plan_t")
local stats_ct   = ffi.typeof("luaTL_pool_stats_t")
local float_ct   = ffi.typeof("float[?]")
local u64_1      = ffi.new("uint64_t[1]")
local u64_2      = ffi.new("uint64_t[1]")
local scratch_plan  = plan_ct()
local scratch_stats = stats_ct()

-- ---------------------------------------------------------------------
--  Error handling
-- ---------------------------------------------------------------------
local function last_error()
    local msg = ffi.string(C.luaTL_get_last_error())
    C.luaTL_clear_error()
    if msg == "" then msg = "unspecified backend error" end
    return msg
end
luaTL.last_error = last_error

local function ck(status, ctx)
    if status ~= 0 then
        error(string.format("luaTL[%s]: %s (%s)", ctx or "?",
              last_error(), STATUS_NAME[status] or ("status " .. tostring(status))), 3)
    end
    return status
end
luaTL.check = ck

-- ---------------------------------------------------------------------
--  Runtime control
-- ---------------------------------------------------------------------
luaTL.devinfo = nil
luaTL.device  = "<uninitialized>"

function luaTL.init(device, opts)
    opts = opts or {}
    C.luaTL_clear_error()
    local n = C.luaTL_init(device or 0)
    if n == 0 then
        error("luaTL.init failed: " .. last_error(), 2)
    end
    C.luaTL_set_verbose(opts.verbose and 1 or 0)

    luaTL.device  = ffi.string(C.luaTL_device_name())
    luaTL.devinfo = devinfo_ct()
    ck(C.luaTL_get_devinfo(-1, luaTL.devinfo), "get_devinfo")
    luaTL.core_version = ffi.string(C.luaTL_version())
    luaTL.pool = C.luaTL_default_pool()
    return n, luaTL.device
end

function luaTL.shutdown() C.luaTL_shutdown() end
function luaTL.sync() ck(C.luaTL_sync(), "sync") end
function luaTL.set_verbose(on) C.luaTL_set_verbose(on and 1 or 0) end

function luaTL.mem_info()
    ck(C.luaTL_mem_info(u64_1, u64_2), "mem_info")
    return tonumber(u64_1[0]), tonumber(u64_2[0])
end

function luaTL.hardware()
    local d = luaTL.devinfo
    if not d then return nil, "call luaTL.init() first" end
    return {
        name              = ffi.string(d.name),
        sm                = string.format("sm_%d%d", d.major, d.minor),
        major             = d.major,
        minor             = d.minor,
        multiprocessors   = d.mp_count,
        cores_per_sm      = d.cores_per_mp,
        cuda_cores        = d.mp_count * d.cores_per_mp,
        warp_size         = d.warp_size,
        max_threads_block = d.max_threads_per_block,
        max_threads_sm    = d.max_threads_per_mp,
        shared_per_block  = tonumber(d.shared_mem_per_block),
        shared_per_sm     = tonumber(d.shared_mem_per_mp),
        total_vram        = tonumber(d.total_global_mem),
        free_vram         = tonumber(d.free_global_mem),
        l2_cache          = d.l2_cache_size,
        bandwidth_gbs     = d.peak_mem_bandwidth_gbs,
        gflops_fp32       = d.peak_fp32_gflops,
        gflops_fp16       = d.peak_fp16_gflops,
        tensor_cores      = d.has_tensor_cores ~= 0,
        fp16              = d.has_fp16 ~= 0,
        bf16              = d.has_bf16 ~= 0,
    }
end

-- ---------------------------------------------------------------------
--  Pool
-- ---------------------------------------------------------------------
function luaTL.pool_stats(pool)
    ck(C.luaTL_pool_get_stats(pool or nil, scratch_stats), "pool_stats")
    local s = scratch_stats
    return {
        reserved   = tonumber(s.bytes_reserved),
        in_use     = tonumber(s.bytes_in_use),
        peak       = tonumber(s.peak_in_use),
        blocks     = tonumber(s.block_count),
        free_blocks= tonumber(s.free_block_count),
        allocs     = tonumber(s.n_alloc),
        frees      = tonumber(s.n_free),
        cuda_malloc= tonumber(s.n_cuda_malloc),
        cuda_free  = tonumber(s.n_cuda_free),
        cache_hit  = tonumber(s.n_cache_hit),
        cache_miss = tonumber(s.n_cache_miss),
        trims      = tonumber(s.n_trim),
    }
end

function luaTL.pool_trim(pool) C.luaTL_pool_trim(pool or nil) end

-- ---------------------------------------------------------------------
--  Arena (bump allocator; Lua bumps `offset` with zero FFI calls)
-- ---------------------------------------------------------------------
local Arena = {}
Arena.__index = Arena

function luaTL.Arena(bytes, pool)
    local a = C.luaTL_arena_create(pool or nil, bytes)
    if a == nil then error("luaTL.Arena: " .. last_error(), 2) end
    return setmetatable({ h = ffi.gc(a, C.luaTL_arena_destroy) }, Arena)
end

function Arena:reset()  self.h.offset = 0 return self end
function Arena:used()   return tonumber(self.h.offset) end
function Arena:peak()   return tonumber(self.h.peak) end
function Arena:capacity() return tonumber(self.h.capacity) end

--- Fast path: pure Lua pointer bump, no FFI transition at all.
function Arena:raw(bytes)
    local h   = self.h
    local al  = 256
    local off = bit.band(tonumber(h.offset) + al - 1, bit.bnot(al - 1))
    if off + bytes > tonumber(h.capacity) then
        error("luaTL.Arena: exhausted (" .. off + bytes .. " > " ..
              tonumber(h.capacity) .. " bytes)", 2)
    end
    h.offset = off + bytes
    if h.offset > h.peak then h.peak = h.offset end
    return h.base + off
end

--- Allocate a tensor whose storage lives in the arena (never freed
--- individually; reclaimed wholesale by Arena:reset()).
function Arena:tensor(rows, cols, dtype)
    dtype = dtype or luaTL.F32
    local n     = rows * cols
    local bytes = n * luaTL.dtype_size[dtype]
    local t     = tensor_ct()
    ck(C.luaTL_tensor_wrap(t, self:raw(bytes), rows, cols, dtype), "arena.tensor")
    return t
end

luaTL.ArenaClass = Arena

-- ---------------------------------------------------------------------
--  Tensor : ffi.metatype over luaTL_tensor_t  (no Lua tables, no hash)
-- ---------------------------------------------------------------------
local Tensor = {}

local function dt_of(x)
    if x == nil then return luaTL.F32 end
    if type(x) == "number" then return x end
    local d = luaTL.dtype[x]
    if d == nil then error("luaTL: unknown dtype '" .. tostring(x) .. "'", 3) end
    return d
end
luaTL.dt_of = dt_of

--- Allocate an uninitialized device tensor.
function luaTL.new(rows, cols, dtype, pool)
    rows = math.floor(rows or 1)
    cols = math.floor(cols or 1)
    if rows <= 0 or cols <= 0 then
        error("luaTL.new: shape must be positive, got " .. rows .. "x" .. cols, 2)
    end
    local t = tensor_ct()
    ck(C.luaTL_tensor_init(t, rows, cols, dt_of(dtype), pool or nil), "new")
    return t
end

function luaTL.zeros(rows, cols, dtype, pool)
    local t = luaTL.new(rows, cols, dtype, pool)
    ck(C.luaTL_tensor_zero(t), "zeros")
    return t
end

--- Wrap an existing device pointer without taking ownership.
function luaTL.wrap(ptr, rows, cols, dtype)
    local t = tensor_ct()
    ck(C.luaTL_tensor_wrap(t, ptr, rows, cols, dt_of(dtype)), "wrap")
    return t
end

--- Build from a flat or nested Lua table (host staging is transient).
function luaTL.from_table(data, shape, dtype)
    if type(data) ~= "table" then
        error("luaTL.from_table expects a table", 2)
    end
    local flat, rows, cols
    if type(data[1]) == "table" then
        rows, cols = #data, #data[1]
        flat = {}
        local k = 0
        for r = 1, rows do
            local row = data[r]
            if #row ~= cols then
                error("luaTL.from_table: ragged row " .. r, 2)
            end
            for c = 1, cols do k = k + 1 flat[k] = row[c] end
        end
    else
        flat = data
        if shape then rows, cols = shape[1], shape[2]
        else rows, cols = 1, #data end
    end
    local n = rows * cols
    if n ~= #flat then
        error(string.format("luaTL.from_table: shape %dx%d != %d elements",
                            rows, cols, #flat), 2)
    end
    local buf = float_ct(n)
    for i = 1, n do buf[i - 1] = flat[i] end
    local t = luaTL.new(rows, cols, dtype)
    ck(C.luaTL_tensor_upload(t, buf, n), "from_table")
    return t
end

function luaTL.random(rows, cols, scale, dtype)
    scale = scale or 1.0
    local n   = rows * cols
    local buf = float_ct(n)
    for i = 0, n - 1 do buf[i] = (math.random() * 2.0 - 1.0) * scale end
    local t = luaTL.new(rows, cols, dtype)
    ck(C.luaTL_tensor_upload(t, buf, n), "random")
    return t
end

--- Xavier / Glorot uniform initialisation.
function luaTL.xavier(rows, cols, dtype)
    return luaTL.random(rows, cols, math.sqrt(6.0 / (rows + cols)), dtype)
end

-- ---- introspection --------------------------------------------------
function Tensor:shape()    return self.rows, self.cols end
function Tensor:numel()    return tonumber(self.nelem) end
function Tensor:bytes()    return tonumber(self.nbytes) end
function Tensor:dtype_name() return luaTL.dtype_name[self.dtype] end
function Tensor:ptr()      return self.data end
function Tensor:is_null()  return self.data == nil end

-- ---- lifetime -------------------------------------------------------
function Tensor:free()
    C.luaTL_tensor_release(self)
    return self
end

function Tensor:zero()
    ck(C.luaTL_tensor_zero(self), "zero")
    return self
end

--- Non-owning row-slice view.  `rows`/`cols` default to the remainder.
function Tensor:view(row_off, rows, cols)
    local v = tensor_ct()
    ck(C.luaTL_tensor_view(v, self, row_off or 0, rows or 0, cols or 0), "view")
    return v
end

function Tensor:reshape(rows, cols)
    if rows * cols ~= tonumber(self.nelem) then
        error(string.format("luaTL:reshape %dx%d != %d elements",
                            rows, cols, tonumber(self.nelem)), 2)
    end
    self.rows, self.cols = rows, cols
    return self
end

function Tensor:clone()
    local t = luaTL.new(self.rows, self.cols, self.dtype)
    ck(C.luaTL_tensor_copy(self, t), "clone")
    return t
end

function Tensor:copy_from(src)
    ck(C.luaTL_tensor_copy(src, self), "copy_from")
    return self
end

function Tensor:cast(dtype)
    local t = luaTL.new(self.rows, self.cols, dt_of(dtype))
    ck(C.luaTL_tensor_cast(self, t), "cast")
    return t
end

-- ---- host transfer --------------------------------------------------
function Tensor:upload(host_buf, n)
    ck(C.luaTL_tensor_upload(self, host_buf, n or 0), "upload")
    return self
end

function Tensor:download(host_buf, n)
    ck(C.luaTL_tensor_download(self, host_buf, n or 0), "download")
    return host_buf
end

function Tensor:tohost()
    local n   = tonumber(self.nelem)
    local buf = float_ct(n)
    ck(C.luaTL_tensor_download(self, buf, n), "tohost")
    return buf, n
end

function Tensor:totable()
    local buf = self:tohost()
    local out, rows, cols = {}, self.rows, self.cols
    for r = 1, rows do
        local row = {}
        local base = (r - 1) * cols
        for c = 1, cols do row[c] = buf[base + c - 1] end
        out[r] = row
    end
    return out
end

function Tensor:toflat()
    local buf, n = self:tohost()
    local out = {}
    for i = 1, n do out[i] = buf[i - 1] end
    return out
end

function Tensor:get(r, c)
    local buf = float_ct(1)
    local off = ((r - 1) * self.cols + (c - 1))
    local view = tensor_ct()
    ck(C.luaTL_tensor_wrap(view, ffi.cast("char*", self.data) +
        off * luaTL.dtype_size[self.dtype], 1, 1, self.dtype), "get")
    ck(C.luaTL_tensor_download(view, buf, 1), "get")
    return buf[0]
end

function Tensor:set(r, c, v)
    local buf = float_ct(1)
    buf[0] = v
    local off = ((r - 1) * self.cols + (c - 1))
    local view = tensor_ct()
    ck(C.luaTL_tensor_wrap(view, ffi.cast("char*", self.data) +
        off * luaTL.dtype_size[self.dtype], 1, 1, self.dtype), "set")
    ck(C.luaTL_tensor_upload(view, buf, 1), "set")
    return self
end

-- ---- math : allocating variants ------------------------------------
function Tensor:matmul(B, opts)
    opts = opts or {}
    if self.cols ~= B.rows then
        error(string.format("luaTL:matmul shape mismatch (%dx%d)*(%dx%d)",
                            self.rows, self.cols, B.rows, B.cols), 2)
    end
    local out = luaTL.new(self.rows, B.cols, self.dtype)
    ck(C.luaTL_gemm(self.data, B.data, out.data,
                    opts.bias and opts.bias.data or nil,
                    opts.gamma and opts.gamma.data or nil, nil,
                    self.rows, B.cols, self.cols, self.dtype,
                    opts.alpha or 1.0, 0.0, opts.eps or 1e-5,
                    opts.pre or 0, opts.act or 0), "matmul")
    return out
end

--- In place: self = act(alpha*(pre(A)@B) + bias) + beta*self
function Tensor:matmul_inplace(A, B, opts)
    opts = opts or {}
    if A.cols ~= B.rows or A.rows ~= self.rows or B.cols ~= self.cols then
        error("luaTL:matmul_inplace shape mismatch", 2)
    end
    ck(C.luaTL_gemm(A.data, B.data, self.data,
                    opts.bias and opts.bias.data or nil,
                    opts.gamma and opts.gamma.data or nil, nil,
                    A.rows, B.cols, A.cols, self.dtype,
                    opts.alpha or 1.0, opts.beta or 0.0, opts.eps or 1e-5,
                    opts.pre or 0, opts.act or 0), "matmul_inplace")
    return self
end

function Tensor:fused_rmsnorm_matmul(W, gamma, bias, eps)
    local out = luaTL.new(self.rows, W.cols, self.dtype)
    ck(C.luaTL_gemm(self.data, W.data, out.data,
                    bias and bias.data or nil,
                    gamma and gamma.data or nil, nil,
                    self.rows, W.cols, self.cols, self.dtype,
                    1.0, 0.0, eps or 1e-5, luaTL.PRE.RMSNORM, 0),
       "fused_rmsnorm_matmul")
    return out
end

function Tensor:fused_gelu_matmul(W, bias)
    local out = luaTL.new(self.rows, W.cols, self.dtype)
    ck(C.luaTL_gemm(self.data, W.data, out.data, bias and bias.data or nil,
                    nil, nil, self.rows, W.cols, self.cols, self.dtype,
                    1.0, 0.0, 1e-5, luaTL.PRE.GELU, 0), "fused_gelu_matmul")
    return out
end

function Tensor:fused_silu_matmul(W, bias)
    local out = luaTL.new(self.rows, W.cols, self.dtype)
    ck(C.luaTL_gemm(self.data, W.data, out.data, bias and bias.data or nil,
                    nil, nil, self.rows, W.cols, self.cols, self.dtype,
                    1.0, 0.0, 1e-5, luaTL.PRE.SILU, 0), "fused_silu_matmul")
    return out
end

function Tensor:add(B)
    local out = luaTL.new(self.rows, self.cols, self.dtype)
    ck(C.luaTL_elementwise(self.data, B.data, out.data, self.nelem,
                           self.dtype, luaTL.EW.ADD, 1.0, 1.0), "add")
    return out
end

--- self += alpha * B  (no allocation whatsoever)
function Tensor:add_inplace(B, alpha)
    ck(C.luaTL_elementwise(self.data, B.data, self.data, self.nelem,
                           self.dtype, luaTL.EW.AXPY, alpha or 1.0, 0.0),
       "add_inplace")
    return self
end

function Tensor:mul_inplace(B)
    ck(C.luaTL_elementwise(self.data, B.data, self.data, self.nelem,
                           self.dtype, luaTL.EW.MUL, 1.0, 0.0), "mul_inplace")
    return self
end

function Tensor:scale_inplace(alpha, beta)
    ck(C.luaTL_elementwise(self.data, nil, self.data, self.nelem,
                           self.dtype, luaTL.EW.SCALE, alpha, beta or 0.0),
       "scale_inplace")
    return self
end

function Tensor:fill(v)
    ck(C.luaTL_elementwise(nil, nil, self.data, self.nelem,
                           self.dtype, luaTL.EW.FILL, v, 0.0), "fill")
    return self
end

function Tensor:apply_inplace(op, alpha, beta)
    ck(C.luaTL_elementwise(self.data, nil, self.data, self.nelem,
                           self.dtype, op, alpha or 0.0, beta or 0.0),
       "apply_inplace")
    return self
end

function Tensor:relu_inplace() return self:apply_inplace(luaTL.EW.RELU) end
function Tensor:gelu_inplace() return self:apply_inplace(luaTL.EW.GELU) end
function Tensor:silu_inplace() return self:apply_inplace(luaTL.EW.SILU) end
function Tensor:clamp_inplace(lo, hi)
    return self:apply_inplace(luaTL.EW.CLAMP, lo, hi)
end

function Tensor:rmsnorm(weight, eps)
    local out = luaTL.new(self.rows, self.cols, self.dtype)
    ck(C.luaTL_rmsnorm(self.data, weight and weight.data or nil, out.data,
                       self.rows, self.cols, eps or 1e-5, self.dtype), "rmsnorm")
    return out
end

function Tensor:rmsnorm_inplace(weight, eps)
    ck(C.luaTL_rmsnorm(self.data, weight and weight.data or nil, self.data,
                       self.rows, self.cols, eps or 1e-5, self.dtype),
       "rmsnorm_inplace")
    return self
end

function Tensor:softmax(opts)
    opts = opts or {}
    local out = luaTL.new(self.rows, self.cols, self.dtype)
    ck(C.luaTL_softmax(self.data, opts.bias and opts.bias.data or nil, out.data,
                       self.rows, self.cols, opts.scale or 1.0,
                       opts.causal and 1 or 0, opts.mask_offset or 0,
                       self.dtype), "softmax")
    return out
end

function Tensor:softmax_inplace(opts)
    opts = opts or {}
    ck(C.luaTL_softmax(self.data, opts.bias and opts.bias.data or nil, self.data,
                       self.rows, self.cols, opts.scale or 1.0,
                       opts.causal and 1 or 0, opts.mask_offset or 0,
                       self.dtype), "softmax_inplace")
    return self
end

function Tensor:bcast_add_inplace(vec, alpha)
    ck(C.luaTL_bcast_add(self.data, vec.data, self.data,
                         self.rows, self.cols, alpha or 1.0, self.dtype),
       "bcast_add_inplace")
    return self
end

function Tensor:transpose()
    local out = luaTL.new(self.cols, self.rows, self.dtype)
    ck(C.luaTL_transpose(self.data, out.data, self.rows, self.cols, self.dtype),
       "transpose")
    return out
end

-- ---- operators / printing -------------------------------------------
local function ts(t)
    return string.format("luaTL.Tensor<%s>(%dx%d, %d bytes, %s)",
        luaTL.dtype_name[t.dtype] or "?", t.rows, t.cols,
        tonumber(t.nbytes), t.owns ~= 0 and "owned" or "view")
end

ffi.metatype(tensor_ct, {
    __index    = Tensor,
    __gc       = C.luaTL_tensor_release,
    __tostring = ts,
    __add      = function(a, b) return a:add(b) end,
    __mul      = function(a, b) return a:matmul(b) end,
    __len      = function(t) return tonumber(t.nelem) end,
})

luaTL.Tensor      = luaTL.new
luaTL.TensorClass = Tensor

function Tensor:print(label)
    if label then io.write(label, "\n") end
    io.write(ts(self), "\n")
    local rows = self:totable()
    local rmax = math.min(#rows, 16)
    for r = 1, rmax do
        local row, cmax = rows[r], math.min(self.cols, 12)
        local parts = {}
        for c = 1, cmax do parts[c] = string.format("%9.4f", row[c]) end
        if self.cols > cmax then parts[#parts + 1] = "..." end
        io.write("  ", table.concat(parts, " "), "\n")
    end
    if #rows > rmax then io.write("  ...\n") end
    return self
end

-- ---------------------------------------------------------------------
--  Autotuner
-- ---------------------------------------------------------------------
local function plan_to_table(p)
    return {
        tm = p.tm, tn = p.tn, tk = p.tk,
        block = { p.block_x, p.block_y },
        grid  = { p.grid_x, p.grid_y },
        shmem = p.shmem_bytes,
        kernel_id = p.kernel_id,
        unroll = p.unroll,
        tensor_core = p.use_tensor_core ~= 0,
        cached = p.from_cache ~= 0,
        ms = p.measured_ms,
    }
end

function luaTL.plan(M, N, K, dtype, allow_tc)
    ck(C.luaTL_plan_gemm(M, N, K, dt_of(dtype),
                         allow_tc == false and 0 or 1, scratch_plan), "plan")
    return plan_to_table(scratch_plan)
end

function luaTL.autotune(M, N, K, dtype, iters)
    ck(C.luaTL_autotune_benchmark(M, N, K, dt_of(dtype), iters or 8,
                                  scratch_plan), "autotune")
    return plan_to_table(scratch_plan)
end

function luaTL.set_tile_override(id) C.luaTL_set_tile_override(id or -1) end
function luaTL.tune_reset() C.luaTL_tune_reset() end

-- ---------------------------------------------------------------------
--  Pipeline : queue many GPU ops, run them with ONE FFI call
-- ---------------------------------------------------------------------
local Pipeline = {}
Pipeline.__index = Pipeline
luaTL.PipelineClass = Pipeline

function luaTL.Pipeline(capacity, timing)
    local h = C.luaTL_pipeline_create(capacity or 512, timing and 1 or 0)
    if h == nil then error("luaTL.Pipeline: " .. last_error(), 2) end
    return setmetatable({ h = ffi.gc(h, C.luaTL_pipeline_destroy) }, Pipeline)
end

function Pipeline:reset()
    self.h.count = 0
    self.h.status = 0
    return self
end

--- Reserve the next command slot.  Pure struct arithmetic: LuaJIT
--- compiles this down to a pointer add plus an integer store.
function Pipeline:slot()
    local h = self.h
    local i = h.count
    if i >= h.capacity then
        error("luaTL.Pipeline: queue full (" .. tonumber(h.capacity) .. ")", 3)
    end
    h.count = i + 1
    local c = h.cmds[i]
    c.op, c.dtype, c.act, c.flags = 0, 0, 0, 0
    c.i0, c.i1, c.i2, c.i3 = 0, 0, 0, 0
    c.f0, c.f1, c.f2, c.f3, c.f4, c.f5 = 0, 0, 0, 0, 0, 0
    c.n = 0
    c.p0, c.p1, c.p2, c.p3, c.p4, c.p5 = nil, nil, nil, nil, nil, nil
    return c
end

function Pipeline:gemm(A, B, Cout, opts)
    opts = opts or {}
    local c = self:slot()
    c.op    = luaTL.OP.GEMM
    c.dtype = A.dtype
    c.act   = opts.act or 0
    c.i0, c.i1, c.i2 = A.rows, B.cols, A.cols
    c.i3    = opts.pre or 0
    c.f0    = opts.alpha or 1.0
    c.f1    = opts.beta  or 0.0
    c.f2    = opts.eps   or 1e-5
    c.p0, c.p1, c.p2 = A.data, B.data, Cout.data
    c.p3 = opts.bias     and opts.bias.data     or nil
    c.p4 = opts.gamma    and opts.gamma.data    or nil
    c.p5 = opts.rowscale and opts.rowscale.data or nil
    return self
end

function Pipeline:rmsnorm_matmul(x, W, out, gamma, bias, eps)
    return self:gemm(x, W, out, { pre = luaTL.PRE.RMSNORM,
                                  gamma = gamma, bias = bias, eps = eps })
end

function Pipeline:gelu_matmul(x, W, out, bias)
    return self:gemm(x, W, out, { pre = luaTL.PRE.GELU, bias = bias })
end

function Pipeline:silu_matmul(x, W, out, bias)
    return self:gemm(x, W, out, { pre = luaTL.PRE.SILU, bias = bias })
end

function Pipeline:ew(op, A, B, out, alpha, beta)
    local c = self:slot()
    c.op    = luaTL.OP.EW
    c.dtype = out.dtype
    c.i0    = op
    c.f0    = alpha or 1.0
    c.f1    = beta  or 0.0
    c.n     = out.nelem
    c.p0    = A and A.data or nil
    c.p1    = B and B.data or nil
    c.p2    = out.data
    return self
end

function Pipeline:add(A, B, out, alpha, beta)
    return self:ew(luaTL.EW.ADD, A, B, out, alpha or 1.0, beta or 1.0)
end

function Pipeline:add_inplace(dst, B, alpha)
    return self:ew(luaTL.EW.AXPY, dst, B, dst, alpha or 1.0, 0.0)
end

function Pipeline:rmsnorm(x, w, out, eps)
    local c = self:slot()
    c.op    = luaTL.OP.RMSNORM
    c.dtype = x.dtype
    c.i0, c.i1 = x.rows, x.cols
    c.f0    = eps or 1e-5
    c.p0    = x.data
    c.p1    = w and w.data or nil
    c.p2    = out.data
    return self
end

function Pipeline:softmax(x, out, opts)
    opts = opts or {}
    local c = self:slot()
    c.op    = luaTL.OP.SOFTMAX
    c.dtype = x.dtype
    c.i0, c.i1 = x.rows, x.cols
    c.i2    = opts.causal and 1 or 0
    c.i3    = opts.mask_offset or 0
    c.f0    = opts.scale or 1.0
    c.p0    = x.data
    c.p1    = opts.bias and opts.bias.data or nil
    c.p2    = out.data
    return self
end

function Pipeline:bcast_add(x, vec, out, alpha)
    local c = self:slot()
    c.op    = luaTL.OP.BCAST_ADD
    c.dtype = x.dtype
    c.i0, c.i1 = x.rows, x.cols
    c.f0    = alpha or 1.0
    c.p0, c.p1, c.p2 = x.data, vec and vec.data or nil, out.data
    return self
end

function Pipeline:transpose(x, out)
    local c = self:slot()
    c.op    = luaTL.OP.TRANSPOSE
    c.dtype = x.dtype
    c.i0, c.i1 = x.rows, x.cols
    c.p0, c.p2 = x.data, out.data
    return self
end

function Pipeline:cast(src, dst)
    local c = self:slot()
    c.op = luaTL.OP.CAST
    c.i0, c.i1 = src.dtype, dst.dtype
    c.n  = src.nelem
    c.p0, c.p2 = src.data, dst.data
    return self
end

function Pipeline:adamw(w, g, m, v, w_lp, o)
    o = o or {}
    local c = self:slot()
    c.op    = luaTL.OP.ADAMW
    c.dtype = g.dtype
    c.i0    = o.step or 1
    c.f0    = o.lr    or 1e-3
    c.f1    = o.beta1 or 0.9
    c.f2    = o.beta2 or 0.999
    c.f3    = o.eps   or 1e-8
    c.f4    = o.wd    or 0.0
    c.f5    = o.clip  or 0.0
    c.n     = w.nelem
    c.p0, c.p1, c.p2, c.p3 = w.data, g.data, m.data, v.data
    c.p4 = w_lp and w_lp.data or nil
    return self
end

function Pipeline:zero(t)
    local c = self:slot()
    c.op = luaTL.OP.ZERO
    c.n  = t.nbytes
    c.p2 = t.data
    return self
end

function Pipeline:copy(src, dst)
    local c = self:slot()
    c.op = luaTL.OP.COPY
    c.n  = src.nbytes
    c.p0, c.p2 = src.data, dst.data
    return self
end

function Pipeline:barrier()
    self:slot().op = luaTL.OP.SYNC
    return self
end

function Pipeline:async(on)
    self.h.flags = on and 0 or 1
    return self
end

--- The single FFI transition that executes the whole queue.
function Pipeline:run(keep)
    local rc = C.luaTL_pipeline_run(self.h)
    if rc ~= 0 then
        error("luaTL.Pipeline:run: " .. last_error() ..
              " (" .. (STATUS_NAME[rc] or rc) .. ")", 2)
    end
    if not keep then self.h.count = 0 end
    return self
end

function Pipeline:wait()
    ck(C.luaTL_pipeline_wait(self.h), "pipeline.wait")
    return self
end

function Pipeline:elapsed_ms() return self.h.last_ms end
function Pipeline:size()       return self.h.count end
function Pipeline:stats()
    return { runs = tonumber(self.h.total_runs),
             ops  = tonumber(self.h.total_ops),
             last_ms = self.h.last_ms }
end

-- ---------------------------------------------------------------------
--  Convenience: fused free functions mirroring the C names
-- ---------------------------------------------------------------------
function luaTL.matmul(A, B, opts) return A:matmul(B, opts) end

function luaTL.adamw(w, g, m, v, w_lp, o)
    o = o or {}
    ck(C.luaTL_adamw(ffi.cast("float*", w.data), g.data,
                     ffi.cast("float*", m.data), ffi.cast("float*", v.data),
                     w_lp and w_lp.data or nil, w.nelem, g.dtype,
                     o.lr or 1e-3, o.beta1 or 0.9, o.beta2 or 0.999,
                     o.eps or 1e-8, o.wd or 0.0, o.step or 1, o.clip or 0.0),
       "adamw")
end

--- Quick sanity benchmark: effective TFLOP/s of one GEMM shape.
function luaTL.benchmark_gemm(M, N, K, dtype, iters)
    iters = iters or 20
    dtype = dt_of(dtype)
    local A = luaTL.random(M, K, 0.1, dtype)
    local B = luaTL.random(K, N, 0.1, dtype)
    local Cc = luaTL.zeros(M, N, dtype)
    Cc:matmul_inplace(A, B)             -- warm up
    luaTL.sync()
    local t0 = os.clock()
    for _ = 1, iters do Cc:matmul_inplace(A, B) end
    luaTL.sync()
    local dt = (os.clock() - t0) / iters
    A:free() B:free() Cc:free()
    return { seconds = dt, tflops = (2.0 * M * N * K) / dt / 1e12 }
end

return luaTL