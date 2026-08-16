/*  Build:
 *    Linux :
 *      nvcc -O3 -std=c++14 --shared -Xcompiler -fPIC \
 *           -gencode arch=compute_70,code=sm_70 \
 *           -gencode arch=compute_75,code=sm_75 \
 *           -gencode arch=compute_86,code=sm_86 \
 *           -gencode arch=compute_89,code=sm_89 \
 *           --use_fast_math -lineinfo \
 *           luaTL_core.cu -o luaTL.so
 *
 *    Windows:
 *      nvcc -O3 -std=c++14 --shared -gencode arch=compute_86,code=sm_86 ^
 *           --use_fast_math luaTL_core.cu -o luaTL.dll */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
  #define LUATL_API __declspec(dllexport)
#else
  #define LUATL_API __attribute__((visibility("default")))
#endif

#define LUATL_VERSION_STRING "luaTL-core 2.0.0"

/*  Compile time configuration*/
#define LUATL_BLOCK_1D          256    /* element-wise block size*/
#define LUATL_ROW_THREADS_MAX   1024   /* max threads for a row kernel*/
#define LUATL_ROW_THREADS_MIN   32     /* must stay a multiple of warp*/
#define LUATL_MAX_GRID_1D       65535u
#define LUATL_AUTOTUNE_SLOTS    256
#define LUATL_ERRBUF            1024

/*  Public enumerations (mirrored verbatim in luaTL.lua) */
enum luaTL_dtype_e {
    LUATL_F32  = 0,
    LUATL_F16  = 1,
    LUATL_BF16 = 2
};

enum luaTL_ew_e {
    LUATL_EW_COPY       = 0,
    LUATL_EW_ADD        = 1,   /* c = alpha*a + beta*b            */
    LUATL_EW_SUB        = 2,   /* c = alpha*a - beta*b            */
    LUATL_EW_MUL        = 3,   /* c = alpha*a*b                   */
    LUATL_EW_DIV        = 4,   /* c = alpha*a / b                 */
    LUATL_EW_SCALE      = 5,   /* c = alpha*a + beta              */
    LUATL_EW_AXPY       = 6,   /* c = alpha*a + b                 */
    LUATL_EW_FILL       = 7,   /* c = alpha                       */
    LUATL_EW_RELU       = 8,
    LUATL_EW_GELU       = 9,
    LUATL_EW_SILU       = 10,
    LUATL_EW_TANH       = 11,
    LUATL_EW_SIGMOID    = 12,
    LUATL_EW_EXP        = 13,
    LUATL_EW_SQRT       = 14,
    LUATL_EW_RSQRT      = 15,
    LUATL_EW_NEG        = 16,
    LUATL_EW_RECIP      = 17,
    LUATL_EW_ABS        = 18,
    LUATL_EW_CLAMP      = 19,  /* c = min(max(a,alpha),beta)      */
    LUATL_EW_DRELU      = 20,  /* c = b * (a > 0)                 */
    LUATL_EW_DSILU      = 21,  /* c = b * silu'(a)                */
    LUATL_EW_DGELU      = 22,  /* c = b * gelu'(a)                */
    LUATL_EW__COUNT     = 23
};

enum luaTL_act_e {           /* GEMM epilogue */
    LUATL_ACT_NONE    = 0,
    LUATL_ACT_RELU    = 1,
    LUATL_ACT_GELU    = 2,
    LUATL_ACT_SILU    = 3,
    LUATL_ACT_TANH    = 4,
    LUATL_ACT_SIGMOID = 5
};

enum luaTL_pre_e {           /* GEMM prologue applied to the A operand */
    LUATL_PRE_NONE    = 0,
    LUATL_PRE_RMSNORM = 1,
    LUATL_PRE_GELU    = 2,
    LUATL_PRE_SILU    = 3,
    LUATL_PRE_RELU    = 4
};

enum luaTL_op_e {            /* pipeline command opcodes */
    LUATL_OP_NOP        = 0,
    LUATL_OP_GEMM       = 1,
    LUATL_OP_EW         = 2,
    LUATL_OP_RMSNORM    = 3,
    LUATL_OP_SOFTMAX    = 4,
    LUATL_OP_CAST       = 5,
    LUATL_OP_TRANSPOSE  = 6,
    LUATL_OP_BCAST_ADD  = 7,
    LUATL_OP_ADAMW      = 8,
    LUATL_OP_ZERO       = 9,
    LUATL_OP_COPY       = 10,
    LUATL_OP_SYNC       = 11,
    LUATL_OP__COUNT     = 12
};

enum luaTL_status_e {
    LUATL_OK            =  0,
    LUATL_ERR_NULL      = -1,
    LUATL_ERR_SHAPE     = -2,
    LUATL_ERR_DTYPE     = -3,
    LUATL_ERR_CUDA      = -4,
    LUATL_ERR_OOM       = -5,
    LUATL_ERR_ARG       = -6,
    LUATL_ERR_CAPACITY  = -7,
    LUATL_ERR_UNSUPPORT = -8
};

/* Error plumbing*/

static char g_err[LUATL_ERRBUF]   = {0};
static int  g_has_err             = 0;
static char g_devname[320]        = {0};
static int  g_verbose             = 0;

static void luaTL_seterr(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
    g_has_err = 1;
    if (g_verbose) fprintf(stderr, "[luaTL] %s\n", g_err);
}

static int luaTL_cuda_fail(const char *ctx, cudaError_t e)
{
    luaTL_seterr("%s failed: %s (cuda error %d)",
                 ctx, cudaGetErrorString(e), (int)e);
    return LUATL_ERR_CUDA;
}

#define LUATL_CHECK(ctx, call)                                            \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) return luaTL_cuda_fail((ctx), _e);         \
    } while (0)

/* Non-synchronizing launch validation (used by the async pipeline). */
static int luaTL_launch_async(const char *ctx)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return luaTL_cuda_fail(ctx, e);
    return LUATL_OK;
}

/* Synchronizing launch validation (used by the blocking one-shot API). */
static int luaTL_launch_sync(const char *ctx)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return luaTL_cuda_fail(ctx, e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return luaTL_cuda_fail(ctx, e);
    return LUATL_OK;
}

/*Public POD structures (byte-identical in ffi.cdef)*/

typedef struct {
    int32_t  device;
    int32_t  major;
    int32_t  minor;
    int32_t  mp_count;
    int32_t  warp_size;
    int32_t  max_threads_per_block;
    int32_t  max_threads_per_mp;
    int32_t  max_blocks_per_mp;
    int32_t  regs_per_block;
    int32_t  regs_per_mp;
    int32_t  clock_khz;
    int32_t  memory_clock_khz;
    int32_t  memory_bus_width;
    int32_t  l2_cache_size;
    int32_t  concurrent_kernels;
    int32_t  async_engines;
    int32_t  unified_addressing;
    int32_t  cooperative_launch;
    int32_t  has_tensor_cores;
    int32_t  has_fp16;
    int32_t  has_bf16;
    int32_t  ecc_enabled;
    int32_t  integrated;
    int32_t  cores_per_mp;
    uint64_t shared_mem_per_block;
    uint64_t shared_mem_per_mp;
    uint64_t total_global_mem;
    uint64_t free_global_mem;
    double   peak_mem_bandwidth_gbs;
    double   peak_fp32_gflops;
    double   peak_fp16_gflops;
    char     name[256];
} luaTL_devinfo_t;

typedef struct {
    int32_t tm;             /* tile rows                                 */
    int32_t tn;             /* tile cols                                 */
    int32_t tk;             /* tile depth                                */
    int32_t block_x;
    int32_t block_y;
    int32_t grid_x;
    int32_t grid_y;
    int32_t shmem_bytes;
    int32_t kernel_id;      /* 0=16x16x16 1=32x32x32 2=8x32x32 3=WMMA    */
    int32_t unroll;
    int32_t use_tensor_core;
    int32_t from_cache;     /* 1 if the plan came from the autotune cache*/
    double  measured_ms;    /* -1 when never benchmarked                 */
} luaTL_plan_t;

typedef struct {
    uint64_t bytes_reserved;   /* total VRAM held by the pool            */
    uint64_t bytes_in_use;     /* handed out to live tensors             */
    uint64_t peak_in_use;
    uint64_t block_count;
    uint64_t free_block_count;
    uint64_t n_alloc;
    uint64_t n_free;
    uint64_t n_cuda_malloc;
    uint64_t n_cuda_free;
    uint64_t n_cache_hit;
    uint64_t n_cache_miss;
    uint64_t n_trim;
} luaTL_pool_stats_t;

typedef struct luaTL_pool_s luaTL_pool_t;   /* opaque to Lua */

typedef struct {
    void*    data;        /* device pointer                              */
    uint64_t nelem;       /* element count                               */
    uint64_t nbytes;      /* allocated byte count                        */
    int32_t  rows;
    int32_t  cols;
    int32_t  dtype;
    int32_t  owns;        /* 1 => release() frees `data`                 */
    void*    pool;        /* luaTL_pool_t* or NULL for raw cudaMalloc    */
    int32_t  device;
    int32_t  flags;       /* user defined                                */
} luaTL_tensor_t;

typedef struct {
    char*    base;        /* device base pointer                         */
    uint64_t capacity;
    uint64_t offset;      /* Lua bumps this directly => zero FFI calls   */
    uint64_t peak;
    uint64_t alignment;
    void*    pool;
} luaTL_arena_t;

typedef struct {
    int32_t  op;
    int32_t  dtype;
    int32_t  act;
    int32_t  flags;
    int32_t  i0, i1, i2, i3;
    float    f0, f1, f2, f3, f4, f5;
    uint64_t n;
    void    *p0, *p1, *p2, *p3, *p4, *p5;
} luaTL_cmd_t;

typedef struct {
    luaTL_cmd_t* cmds;
    int32_t      count;      /* written by Lua                           */
    int32_t      capacity;
    int32_t      flags;      /* bit0: synchronize at end of run          */
    int32_t      status;     /* last run status                          */
    void*        stream;     /* cudaStream_t                             */
    void*        ev_start;
    void*        ev_stop;
    double       last_ms;
    uint64_t     total_runs;
    uint64_t     total_ops;
    int32_t      owns_stream;
    int32_t      timing;     /* 1 => record events around the run        */
} luaTL_pipeline_t;

/* Numeric traits — fp32 / fp16 / bf16*/

typedef uint16_t luatl_bf16;

__host__ __device__ __forceinline__ float luatl_bf16_to_f32(luatl_bf16 v)
{
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)v) << 16;
    return c.f;
}

/* Round-to-nearest-even fp32 -> bf16, NaN preserving.  Pure integer math
 * so it needs no sm_80 hardware and is bit-identical on host & device. */
__host__ __device__ __forceinline__ luatl_bf16 luatl_f32_to_bf16(float f)
{
    union { uint32_t u; float f; } c;
    c.f = f;
    uint32_t u = c.u;
    if (((u >> 23) & 0xFFu) == 0xFFu && (u & 0x7FFFFFu) != 0u)
        return (luatl_bf16)0x7FC0u;                  /* quiet NaN */
    uint32_t lsb   = (u >> 16) & 1u;
    uint32_t round = 0x7FFFu + lsb;
    u += round;
    return (luatl_bf16)(u >> 16);
}

template <typename T> struct TAcc;

template <> struct TAcc<float> {
    __host__ __device__ static __forceinline__ float  ld(const float* p, size_t i) { return p[i]; }
    __host__ __device__ static __forceinline__ void   st(float* p, size_t i, float v) { p[i] = v; }
    __host__ __device__ static __forceinline__ float  cvt_from(float v) { return v; }
    __host__ __device__ static __forceinline__ float  cvt_to(float v)   { return v; }
    static __forceinline__ int dtype() { return LUATL_F32; }
};

template <> struct TAcc<__half> {
    __device__ static __forceinline__ float ld(const __half* p, size_t i) { return __half2float(p[i]); }
    __device__ static __forceinline__ void  st(__half* p, size_t i, float v) { p[i] = __float2half(v); }
    __host__ __device__ static __forceinline__ float  cvt_from(__half v) { return __half2float(v); }
    __host__ __device__ static __forceinline__ __half cvt_to(float v)    { return __float2half(v); }
    static __forceinline__ int dtype() { return LUATL_F16; }
};

template <> struct TAcc<luatl_bf16> {
    __host__ __device__ static __forceinline__ float ld(const luatl_bf16* p, size_t i) { return luatl_bf16_to_f32(p[i]); }
    __host__ __device__ static __forceinline__ void  st(luatl_bf16* p, size_t i, float v) { p[i] = luatl_f32_to_bf16(v); }
    __host__ __device__ static __forceinline__ float      cvt_from(luatl_bf16 v) { return luatl_bf16_to_f32(v); }
    __host__ __device__ static __forceinline__ luatl_bf16 cvt_to(float v)        { return luatl_f32_to_bf16(v); }
    static __forceinline__ int dtype() { return LUATL_BF16; }
};

static __forceinline__ size_t luaTL_dtype_size(int dt)
{
    switch (dt) {
        case LUATL_F32:  return 4;
        case LUATL_F16:  return 2;
        case LUATL_BF16: return 2;
        default:         return 0;
    }
}

static __forceinline__ const char* luaTL_dtype_name(int dt)
{
    switch (dt) {
        case LUATL_F32:  return "f32";
        case LUATL_F16:  return "f16";
        case LUATL_BF16: return "bf16";
        default:         return "?";
    }
}

/*Device-side math helpers*/

#define LUATL_GELU_C0 0.7978845608028654f   /* sqrt(2/pi)               */
#define LUATL_GELU_C1 0.044715f

__device__ __forceinline__ float luatl_sigmoid(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float luatl_gelu(float x)
{
    const float x3 = x * x * x;
    const float t  = LUATL_GELU_C0 * (x + LUATL_GELU_C1 * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

__device__ __forceinline__ float luatl_gelu_grad(float x)
{
    const float x3 = x * x * x;
    const float t  = LUATL_GELU_C0 * (x + LUATL_GELU_C1 * x3);
    const float th = tanhf(t);
    const float dt = LUATL_GELU_C0 * (1.0f + 3.0f * LUATL_GELU_C1 * x * x);
    return 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * dt;
}

__device__ __forceinline__ float luatl_silu(float x)
{
    return x * luatl_sigmoid(x);
}

__device__ __forceinline__ float luatl_silu_grad(float x)
{
    const float s = luatl_sigmoid(x);
    return s * (1.0f + x * (1.0f - s));
}

__device__ __forceinline__ float luatl_apply_act(int act, float v)
{
    switch (act) {
        case LUATL_ACT_RELU:    return fmaxf(v, 0.0f);
        case LUATL_ACT_GELU:    return luatl_gelu(v);
        case LUATL_ACT_SILU:    return luatl_silu(v);
        case LUATL_ACT_TANH:    return tanhf(v);
        case LUATL_ACT_SIGMOID: return luatl_sigmoid(v);
        default:                return v;
    }
}

__device__ __forceinline__ float luatl_apply_pre(int pre, float v)
{
    switch (pre) {
        case LUATL_PRE_GELU: return luatl_gelu(v);
        case LUATL_PRE_SILU: return luatl_silu(v);
        case LUATL_PRE_RELU: return fmaxf(v, 0.0f);
        default:             return v;
    }
}

/* warp / block reductions */

__device__ __forceinline__ float luatl_warp_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    return v;
}

__device__ __forceinline__ float luatl_warp_max(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFu, v, o));
    return v;
}

/* `smem` must hold at least 32 floats.  blockDim.x MUST be a multiple of
 * 32 and every thread in the block must reach this call. */
__device__ __forceinline__ float luatl_block_sum(float v, float* smem)
{
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nw   = (blockDim.x + 31) >> 5;

    v = luatl_warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    if (wid == 0) {
        float t = (lane < nw) ? smem[lane] : 0.0f;
        t = luatl_warp_sum(t);
        if (lane == 0) smem[0] = t;
    }
    __syncthreads();
    const float out = smem[0];
    __syncthreads();          /* allow the caller to reuse `smem` */
    return out;
}

__device__ __forceinline__ float luatl_block_max(float v, float* smem)
{
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nw   = (blockDim.x + 31) >> 5;

    v = luatl_warp_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    if (wid == 0) {
        float t = (lane < nw) ? smem[lane] : -INFINITY;
        t = luatl_warp_max(t);
        if (lane == 0) smem[0] = t;
    }
    __syncthreads();
    const float out = smem[0];
    __syncthreads();
    return out;
}

/*Global device state + hardware autotuner tables*/

static int              g_initialized   = 0;
static int              g_device        = 0;
static cudaDeviceProp   g_prop;
static luaTL_devinfo_t  g_devinfo;
static int              g_tile_override = -1;   /* -1 = automatic */

static int luaTL_cores_per_mp(int major, int minor)
{
    switch (major) {
        case 3:  return 192;                                   /* Kepler   */
        case 5:  return 128;                                   /* Maxwell  */
        case 6:  return (minor == 0) ? 64 : 128;               /* Pascal   */
        case 7:  return (minor == 5) ? 64 : 64;                /* Volta/Tu */
        case 8:  return (minor == 0) ? 64 : 128;               /* Ampere   */
        case 9:  return 128;                                   /* Hopper   */
        case 10: return 128;                                   /* Blackwell*/
        case 12: return 128;
        default: return 64;
    }
}

static void luaTL_fill_devinfo(int device, const cudaDeviceProp* p,
                               luaTL_devinfo_t* d)
{
    memset(d, 0, sizeof(*d));
    d->device                = device;
    d->major                 = p->major;
    d->minor                 = p->minor;
    d->mp_count              = p->multiProcessorCount;
    d->warp_size             = p->warpSize;
    d->max_threads_per_block = p->maxThreadsPerBlock;
    d->max_threads_per_mp    = p->maxThreadsPerMultiProcessor;
    d->regs_per_block        = p->regsPerBlock;
#if CUDART_VERSION >= 9000
    d->regs_per_mp           = p->regsPerMultiprocessor;
    d->shared_mem_per_mp     = (uint64_t)p->sharedMemPerMultiprocessor;
#else
    d->regs_per_mp           = p->regsPerBlock;
    d->shared_mem_per_mp     = (uint64_t)p->sharedMemPerBlock;
#endif
    d->clock_khz             = p->clockRate;
    d->memory_clock_khz      = p->memoryClockRate;
    d->memory_bus_width      = p->memoryBusWidth;
    d->l2_cache_size         = p->l2CacheSize;
    d->concurrent_kernels    = p->concurrentKernels;
    d->async_engines         = p->asyncEngineCount;
    d->unified_addressing    = p->unifiedAddressing;
    d->cooperative_launch    = p->cooperativeLaunch;
    d->ecc_enabled           = p->ECCEnabled;
    d->integrated            = p->integrated;
    d->shared_mem_per_block  = (uint64_t)p->sharedMemPerBlock;
    d->total_global_mem      = (uint64_t)p->totalGlobalMem;
    d->has_tensor_cores      = (p->major >= 7) ? 1 : 0;
    d->has_fp16              = (p->major > 5 || (p->major == 5 && p->minor >= 3)) ? 1 : 0;
    d->has_bf16              = (p->major >= 8) ? 1 : 0;
    d->cores_per_mp          = luaTL_cores_per_mp(p->major, p->minor);
    d->max_blocks_per_mp     = (p->maxThreadsPerMultiProcessor > 0)
                             ? (p->maxThreadsPerMultiProcessor / 64) : 16;

    const double mem_ghz = (double)p->memoryClockRate * 1e-6;   /* kHz->GHz */
    d->peak_mem_bandwidth_gbs = mem_ghz * ((double)p->memoryBusWidth / 8.0) * 2.0;

    const double core_ghz = (double)p->clockRate * 1e-6;
    d->peak_fp32_gflops = 2.0 * (double)d->cores_per_mp *
                          (double)p->multiProcessorCount * core_ghz;
    d->peak_fp16_gflops = d->has_tensor_cores
                        ? d->peak_fp32_gflops * 8.0
                        : d->peak_fp32_gflops * 2.0;

    size_t freeb = 0, totb = 0;
    if (cudaMemGetInfo(&freeb, &totb) == cudaSuccess) {
        d->free_global_mem  = (uint64_t)freeb;
        d->total_global_mem = (uint64_t)totb;
    }
    snprintf(d->name, sizeof(d->name), "%s", p->name);
}

/* grid sizing helper */
static __forceinline__ unsigned int luaTL_grid1d(uint64_t n, int block)
{
    uint64_t g = (n + (uint64_t)block - 1) / (uint64_t)block;
    uint64_t cap = (uint64_t)(g_prop.multiProcessorCount > 0
                              ? g_prop.multiProcessorCount : 16) * 64ull;
    if (cap < 64ull) cap = 64ull;
    if (g > cap) g = cap;
    if (g < 1ull) g = 1ull;
    if (g > 2147483647ull) g = 2147483647ull;
    return (unsigned int)g;
}

/* Round threads-per-row up to a warp multiple, clamped to hardware. */
static __forceinline__ int luaTL_row_threads(int cols)
{
    int t = 32;
    while (t < cols && t < LUATL_ROW_THREADS_MAX) t <<= 1;
    if (t > g_prop.maxThreadsPerBlock) t = g_prop.maxThreadsPerBlock;
    t = (t / 32) * 32;
    if (t < LUATL_ROW_THREADS_MIN) t = LUATL_ROW_THREADS_MIN;
    return t;
}

/*Caching device memory pool  (zero-allocation engine)*/

#define LUATL_MIN_SHIFT   8                       /* 256 B smallest class */
#define LUATL_MAX_SHIFT   40                      /* 1 TB largest class   */
#define LUATL_NBUCKETS    (LUATL_MAX_SHIFT - LUATL_MIN_SHIFT + 1)

typedef struct {
    void*    ptr;
    uint64_t bytes;      /* rounded size actually held                    */
    int32_t  bucket;
    int32_t  in_use;
    int32_t  next;       /* next index in the free-list / dead-list chain */
    int32_t  alive;      /* 0 once cudaFree'd by a trim                   */
} luaTL_blk_t;

struct luaTL_pool_s {
    luaTL_blk_t* blks;
    int32_t      nblk;
    int32_t      cap_blk;
    int32_t      dead_head;
    int32_t      head[LUATL_NBUCKETS];

    /* open addressed pointer -> block-index map */
    uintptr_t*   hkey;
    int32_t*     hval;
    int32_t      hcap;
    int32_t      hcount;

    int32_t      device;
    int32_t      _pad;

    luaTL_pool_stats_t st;
};

#define LUATL_HASH_EMPTY  ((uintptr_t)0)
#define LUATL_HASH_TOMB   ((uintptr_t)1)

static __forceinline__ uint64_t luaTL_hash_ptr(uintptr_t p)
{
    uint64_t x = (uint64_t)p;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static int luaTL_hash_init(luaTL_pool_t* P, int cap)
{
    P->hkey = (uintptr_t*)calloc((size_t)cap, sizeof(uintptr_t));
    P->hval = (int32_t*)  calloc((size_t)cap, sizeof(int32_t));
    if (!P->hkey || !P->hval) {
        free(P->hkey); free(P->hval);
        P->hkey = NULL; P->hval = NULL;
        return LUATL_ERR_OOM;
    }
    P->hcap   = cap;
    P->hcount = 0;
    return LUATL_OK;
}

static void luaTL_hash_put_raw(uintptr_t* keys, int32_t* vals, int cap,
                               uintptr_t key, int32_t val)
{
    uint64_t h = luaTL_hash_ptr(key);
    int idx = (int)(h & (uint64_t)(cap - 1));
    for (;;) {
        uintptr_t k = keys[idx];
        if (k == LUATL_HASH_EMPTY || k == LUATL_HASH_TOMB || k == key) {
            keys[idx] = key;
            vals[idx] = val;
            return;
        }
        idx = (idx + 1) & (cap - 1);
    }
}

static int luaTL_hash_grow(luaTL_pool_t* P)
{
    const int ncap = (P->hcap > 0) ? (P->hcap * 2) : 256;
    uintptr_t* nk = (uintptr_t*)calloc((size_t)ncap, sizeof(uintptr_t));
    int32_t*   nv = (int32_t*)  calloc((size_t)ncap, sizeof(int32_t));
    if (!nk || !nv) { free(nk); free(nv); return LUATL_ERR_OOM; }

    for (int i = 0; i < P->hcap; ++i) {
        const uintptr_t k = P->hkey[i];
        if (k != LUATL_HASH_EMPTY && k != LUATL_HASH_TOMB)
            luaTL_hash_put_raw(nk, nv, ncap, k, P->hval[i]);
    }
    free(P->hkey); free(P->hval);
    P->hkey = nk; P->hval = nv; P->hcap = ncap;
    return LUATL_OK;
}

static int luaTL_hash_insert(luaTL_pool_t* P, uintptr_t key, int32_t val)
{
    if (P->hcount * 10 >= P->hcap * 7) {
        int rc = luaTL_hash_grow(P);
        if (rc != LUATL_OK) return rc;
    }
    luaTL_hash_put_raw(P->hkey, P->hval, P->hcap, key, val);
    P->hcount++;
    return LUATL_OK;
}

static int32_t luaTL_hash_find(const luaTL_pool_t* P, uintptr_t key)
{
    if (P->hcap <= 0) return -1;
    uint64_t h = luaTL_hash_ptr(key);
    int idx = (int)(h & (uint64_t)(P->hcap - 1));
    for (int probe = 0; probe < P->hcap; ++probe) {
        const uintptr_t k = P->hkey[idx];
        if (k == LUATL_HASH_EMPTY) return -1;
        if (k == key)              return P->hval[idx];
        idx = (idx + 1) & (P->hcap - 1);
    }
    return -1;
}

static void luaTL_hash_remove(luaTL_pool_t* P, uintptr_t key)
{
    if (P->hcap <= 0) return;
    uint64_t h = luaTL_hash_ptr(key);
    int idx = (int)(h & (uint64_t)(P->hcap - 1));
    for (int probe = 0; probe < P->hcap; ++probe) {
        const uintptr_t k = P->hkey[idx];
        if (k == LUATL_HASH_EMPTY) return;
        if (k == key) {
            P->hkey[idx] = LUATL_HASH_TOMB;
            P->hval[idx] = -1;
            if (P->hcount > 0) P->hcount--;
            return;
        }
        idx = (idx + 1) & (P->hcap - 1);
    }
}

static int luaTL_bucket_of(uint64_t bytes, uint64_t* rounded)
{
    int      s = LUATL_MIN_SHIFT;
    uint64_t r = 1ull << s;
    while (r < bytes && s < LUATL_MAX_SHIFT) { ++s; r <<= 1; }
    *rounded = r;
    return s - LUATL_MIN_SHIFT;
}

static int32_t luaTL_pool_new_slot(luaTL_pool_t* P)
{
    if (P->dead_head >= 0) {
        int32_t idx = P->dead_head;
        P->dead_head = P->blks[idx].next;
        memset(&P->blks[idx], 0, sizeof(luaTL_blk_t));
        P->blks[idx].next = -1;
        return idx;
    }
    if (P->nblk >= P->cap_blk) {
        int32_t ncap = (P->cap_blk > 0) ? P->cap_blk * 2 : 128;
        luaTL_blk_t* nb = (luaTL_blk_t*)realloc(P->blks,
                                (size_t)ncap * sizeof(luaTL_blk_t));
        if (!nb) return -1;
        P->blks    = nb;
        P->cap_blk = ncap;
    }
    int32_t idx = P->nblk++;
    memset(&P->blks[idx], 0, sizeof(luaTL_blk_t));
    P->blks[idx].next = -1;
    return idx;
}

static void luaTL_pool_trim_internal(luaTL_pool_t* P)
{
    for (int b = 0; b < LUATL_NBUCKETS; ++b) {
        int32_t idx = P->head[b];
        while (idx >= 0) {
            const int32_t nxt = P->blks[idx].next;
            if (P->blks[idx].ptr) {
                cudaFree(P->blks[idx].ptr);
                P->st.n_cuda_free++;
                if (P->st.bytes_reserved >= P->blks[idx].bytes)
                    P->st.bytes_reserved -= P->blks[idx].bytes;
                luaTL_hash_remove(P, (uintptr_t)P->blks[idx].ptr);
            }
            P->blks[idx].ptr   = NULL;
            P->blks[idx].alive = 0;
            P->blks[idx].bytes = 0;
            P->blks[idx].next  = P->dead_head;
            P->dead_head       = idx;
            if (P->st.block_count > 0) P->st.block_count--;
            if (P->st.free_block_count > 0) P->st.free_block_count--;
            idx = nxt;
        }
        P->head[b] = -1;
    }
    P->st.n_trim++;
    cudaGetLastError();
}

static luaTL_pool_t* luaTL_pool_new(int device)
{
    luaTL_pool_t* P = (luaTL_pool_t*)calloc(1, sizeof(luaTL_pool_t));
    if (!P) return NULL;
    for (int i = 0; i < LUATL_NBUCKETS; ++i) P->head[i] = -1;
    P->dead_head = -1;
    P->device    = device;
    if (luaTL_hash_init(P, 1024) != LUATL_OK) { free(P); return NULL; }
    return P;
}

static void* luaTL_pool_alloc_internal(luaTL_pool_t* P, uint64_t bytes)
{
    if (!P)          { luaTL_seterr("pool_alloc: null pool");  return NULL; }
    if (bytes == 0)  bytes = 1;

    uint64_t rb;
    const int b = luaTL_bucket_of(bytes, &rb);

    P->st.n_alloc++;

    int32_t idx = P->head[b];
    if (idx >= 0) {
        P->head[b]            = P->blks[idx].next;
        P->blks[idx].next     = -1;
        P->blks[idx].in_use   = 1;
        P->st.n_cache_hit++;
        if (P->st.free_block_count > 0) P->st.free_block_count--;
        P->st.bytes_in_use += P->blks[idx].bytes;
        if (P->st.bytes_in_use > P->st.peak_in_use)
            P->st.peak_in_use = P->st.bytes_in_use;
        return P->blks[idx].ptr;
    }

    P->st.n_cache_miss++;

    void* ptr = NULL;
    cudaError_t e = cudaMalloc(&ptr, (size_t)rb);
    if (e != cudaSuccess) {
        /* Out of VRAM: release every cached-but-idle block and retry once. */
        cudaGetLastError();
        luaTL_pool_trim_internal(P);
        e = cudaMalloc(&ptr, (size_t)rb);
    }
    if (e != cudaSuccess) {
        cudaGetLastError();
        luaTL_seterr("pool_alloc: cudaMalloc(%llu bytes) failed: %s",
                     (unsigned long long)rb, cudaGetErrorString(e));
        return NULL;
    }

    idx = luaTL_pool_new_slot(P);
    if (idx < 0) {
        cudaFree(ptr);
        luaTL_seterr("pool_alloc: host block table out of memory");
        return NULL;
    }

    P->blks[idx].ptr    = ptr;
    P->blks[idx].bytes  = rb;
    P->blks[idx].bucket = b;
    P->blks[idx].in_use = 1;
    P->blks[idx].alive  = 1;
    P->blks[idx].next   = -1;

    if (luaTL_hash_insert(P, (uintptr_t)ptr, idx) != LUATL_OK) {
        cudaFree(ptr);
        P->blks[idx].alive = 0;
        P->blks[idx].ptr   = NULL;
        P->blks[idx].next  = P->dead_head;
        P->dead_head       = idx;
        luaTL_seterr("pool_alloc: hash table out of memory");
        return NULL;
    }

    P->st.n_cuda_malloc++;
    P->st.block_count++;
    P->st.bytes_reserved += rb;
    P->st.bytes_in_use   += rb;
    if (P->st.bytes_in_use > P->st.peak_in_use)
        P->st.peak_in_use = P->st.bytes_in_use;
    return ptr;
}

static void luaTL_pool_free_internal(luaTL_pool_t* P, void* ptr)
{
    if (!ptr) return;
    if (!P) { cudaFree(ptr); return; }

    const int32_t idx = luaTL_hash_find(P, (uintptr_t)ptr);
    if (idx < 0 || idx >= P->nblk) {
        /* Foreign pointer (raw cudaMalloc): free it directly. */
        cudaFree(ptr);
        cudaGetLastError();
        return;
    }
    if (!P->blks[idx].in_use) return;    /* double free guard */

    const int b = P->blks[idx].bucket;
    P->blks[idx].in_use = 0;
    P->blks[idx].next   = P->head[b];
    P->head[b]          = idx;

    P->st.n_free++;
    P->st.free_block_count++;
    if (P->st.bytes_in_use >= P->blks[idx].bytes)
        P->st.bytes_in_use -= P->blks[idx].bytes;
}

static luaTL_pool_t* g_pool = NULL;

/*Host staging buffer (dtype conversion on upload)*/

static void*  g_stage      = NULL;
static size_t g_stage_size = 0;

static void* luaTL_stage(size_t bytes)
{
    if (bytes <= g_stage_size && g_stage) return g_stage;
    void* np = realloc(g_stage, bytes);
    if (!np) { luaTL_seterr("host staging buffer OOM (%zu bytes)", bytes); return NULL; }
    g_stage      = np;
    g_stage_size = bytes;
    return g_stage;
}

/*Element-wise kernels*/

__device__ __forceinline__ float luatl_ew_apply(int op, float a, float b,
                                                float alpha, float beta)
{
    switch (op) {
        case LUATL_EW_COPY:    return a;
        case LUATL_EW_ADD:     return alpha * a + beta * b;
        case LUATL_EW_SUB:     return alpha * a - beta * b;
        case LUATL_EW_MUL:     return alpha * a * b;
        case LUATL_EW_DIV:     return alpha * a / (b + 1e-20f);
        case LUATL_EW_SCALE:   return alpha * a + beta;
        case LUATL_EW_AXPY:    return alpha * a + b;
        case LUATL_EW_FILL:    return alpha;
        case LUATL_EW_RELU:    return fmaxf(a, 0.0f);
        case LUATL_EW_GELU:    return luatl_gelu(a);
        case LUATL_EW_SILU:    return luatl_silu(a);
        case LUATL_EW_TANH:    return tanhf(a);
        case LUATL_EW_SIGMOID: return luatl_sigmoid(a);
        case LUATL_EW_EXP:     return __expf(a);
        case LUATL_EW_SQRT:    return sqrtf(fmaxf(a, 0.0f));
        case LUATL_EW_RSQRT:   return rsqrtf(fmaxf(a, 1e-20f));
        case LUATL_EW_NEG:     return -a;
        case LUATL_EW_RECIP:   return 1.0f / (a + 1e-20f);
        case LUATL_EW_ABS:     return fabsf(a);
        case LUATL_EW_CLAMP:   return fminf(fmaxf(a, alpha), beta);
        case LUATL_EW_DRELU:   return (a > 0.0f) ? b : 0.0f;
        case LUATL_EW_DSILU:   return b * luatl_silu_grad(a);
        case LUATL_EW_DGELU:   return b * luatl_gelu_grad(a);
        default:               return a;
    }
}

/* B may be NULL for unary operators. */
template <typename T>
__global__ void luaTL_ew_kernel(const T* __restrict__ A,
                                const T* __restrict__ B,
                                T*       __restrict__ C,
                                uint64_t n, int op, float alpha, float beta)
{
    uint64_t idx    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        const float a = (A != NULL) ? TAcc<T>::ld(A, (size_t)idx) : 0.0f;
        const float b = (B != NULL) ? TAcc<T>::ld(B, (size_t)idx) : 0.0f;
        TAcc<T>::st(C, (size_t)idx, luatl_ew_apply(op, a, b, alpha, beta));
    }
}

/* Vectorised half2 fast path for the bandwidth bound binary operators. */
__global__ void luaTL_ew_half2_kernel(const __half2* __restrict__ A,
                                      const __half2* __restrict__ B,
                                      __half2*       __restrict__ C,
                                      uint64_t n2, int op,
                                      float alpha, float beta)
{
    uint64_t idx    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    const __half2 ha = __float2half2_rn(alpha);
    const __half2 hb = __float2half2_rn(beta);
    for (; idx < n2; idx += stride) {
        const __half2 a = A[idx];
        switch (op) {
            case LUATL_EW_ADD:
                C[idx] = __hadd2(__hmul2(ha, a), __hmul2(hb, B[idx]));
                break;
            case LUATL_EW_SUB:
                C[idx] = __hsub2(__hmul2(ha, a), __hmul2(hb, B[idx]));
                break;
            case LUATL_EW_MUL:
                C[idx] = __hmul2(ha, __hmul2(a, B[idx]));
                break;
            case LUATL_EW_AXPY:
                C[idx] = __hadd2(__hmul2(ha, a), B[idx]);
                break;
            case LUATL_EW_SCALE:
                C[idx] = __hadd2(__hmul2(ha, a), hb);
                break;
            default:
                C[idx] = a;
                break;
        }
    }
#else
    /* Scalar fallback for pre-Pascal hardware without half2 ALUs. */
    for (; idx < n2; idx += stride) {
        const __half2 av = A[idx];
        const __half2 bv = (B != NULL) ? B[idx] : __float2half2_rn(0.0f);
        const float a0 = __low2float(av),  a1 = __high2float(av);
        const float b0 = __low2float(bv),  b1 = __high2float(bv);
        C[idx] = __floats2half2_rn(luatl_ew_apply(op, a0, b0, alpha, beta),
                                   luatl_ew_apply(op, a1, b1, alpha, beta));
    }
#endif
}

/* dtype conversion */
template <typename TS, typename TD>
__global__ void luaTL_cast_kernel(const TS* __restrict__ src,
                                  TD*       __restrict__ dst,
                                  uint64_t n)
{
    uint64_t idx    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride)
        TAcc<TD>::st(dst, (size_t)idx, TAcc<TS>::ld(src, (size_t)idx));
}

/* row-broadcast add: out[r][c] = x[r][c] + v[c] */
template <typename T>
__global__ void luaTL_bcast_add_kernel(const T* __restrict__ x,
                                       const T* __restrict__ v,
                                       T*       __restrict__ out,
                                       int rows, int cols, float alpha)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)cols;
    uint64_t idx     = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        const int c = (int)(idx % (uint64_t)cols);
        const float a = TAcc<T>::ld(x, (size_t)idx);
        const float b = (v != NULL) ? TAcc<T>::ld(v, (size_t)c) : 0.0f;
        TAcc<T>::st(out, (size_t)idx, a + alpha * b);
    }
}

/* tiled transpose */
template <typename T>
__global__ void luaTL_transpose_kernel(const T* __restrict__ in,
                                       T*       __restrict__ out,
                                       int rows, int cols)
{
    __shared__ float tile[32][33];

    const int x = blockIdx.x * 32 + threadIdx.x;   /* column in `in`  */
    const int y = blockIdx.y * 32 + threadIdx.y;   /* row    in `in`  */

    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int yy = y + j;
        tile[threadIdx.y + j][threadIdx.x] =
            (yy < rows && x < cols) ? TAcc<T>::ld(in, (size_t)yy * cols + x)
                                    : 0.0f;
    }
    __syncthreads();

    const int tx = blockIdx.y * 32 + threadIdx.x;  /* column in `out` */
    const int ty = blockIdx.x * 32 + threadIdx.y;  /* row    in `out` */

    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int tyy = ty + j;
        if (tyy < cols && tx < rows)
            TAcc<T>::st(out, (size_t)tyy * rows + tx,
                        tile[threadIdx.x][threadIdx.y + j]);
    }
}

/*RMSNorm / Softmax kernels (fp32 / fp16 / bf16)*/

/* Per-row inverse RMS, written as fp32 so the fused GEMM can consume it. */
template <typename T>
__global__ void luaTL_rms_scale_kernel(const T* __restrict__ x,
                                       float*   __restrict__ scale,
                                       int rows, int cols, float eps)
{
    __shared__ float red[32];
    const int row = blockIdx.x;
    const T*  xr  = x + (size_t)row * cols;

    float local = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = TAcc<T>::ld(xr, (size_t)i);
        local = fmaf(v, v, local);
    }
    const float total = luatl_block_sum(local, red);

    if (threadIdx.x == 0)
        scale[row] = rsqrtf(total / (float)cols + eps);
}

/* out[r][i] = x[r][i] * rsqrt(mean(x[r]^2)+eps) * w[i]   (w may be NULL)
 * Safe for out == x (true in-place).                                   */
template <typename T>
__global__ void luaTL_rmsnorm_kernel(const T* __restrict__ x,
                                     const T* __restrict__ w,
                                     T*       __restrict__ out,
                                     int rows, int cols, float eps)
{
    __shared__ float red[32];
    const int row = blockIdx.x;
    const T*  xr  = x   + (size_t)row * cols;
    T*        orw = out + (size_t)row * cols;

    float local = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = TAcc<T>::ld(xr, (size_t)i);
        local = fmaf(v, v, local);
    }
    const float total = luatl_block_sum(local, red);
    const float scale = rsqrtf(total / (float)cols + eps);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float g = (w != NULL) ? TAcc<T>::ld(w, (size_t)i) : 1.0f;
        TAcc<T>::st(orw, (size_t)i, TAcc<T>::ld(xr, (size_t)i) * scale * g);
    }
}

/* Fused bias-add + scale + optional causal mask + numerically stable
 * row-wise softmax.  Safe for out == x.
 *   pre[r][c] = x[r][c]*scale + bias[c]
 *   causal    : positions with c > r + mask_offset are set to -inf      */
template <typename T>
__global__ void luaTL_softmax_kernel(const T* __restrict__ x,
                                     const T* __restrict__ bias,
                                     T*       __restrict__ out,
                                     int rows, int cols,
                                     float scale, int causal, int mask_offset)
{
    __shared__ float red[32];
    const int row = blockIdx.x;
    const T*  xr  = x   + (size_t)row * cols;
    T*        orw = out + (size_t)row * cols;

    const int limit = causal ? (row + mask_offset) : (cols - 1);

    /* ---- pass 1 : row maximum ---- */
    float lmax = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i > limit) continue;
        float v = TAcc<T>::ld(xr, (size_t)i) * scale;
        if (bias != NULL) v += TAcc<T>::ld(bias, (size_t)i);
        lmax = fmaxf(lmax, v);
    }
    const float rmax = luatl_block_max(lmax, red);
    const float base = (rmax == -INFINITY) ? 0.0f : rmax;

    /* ---- pass 2 : exponentials + sum ---- */
    float lsum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = 0.0f;
        if (i <= limit) {
            float v = TAcc<T>::ld(xr, (size_t)i) * scale;
            if (bias != NULL) v += TAcc<T>::ld(bias, (size_t)i);
            e = __expf(v - base);
        }
        TAcc<T>::st(orw, (size_t)i, e);
        lsum += e;
    }
    const float total = luatl_block_sum(lsum, red);
    const float inv   = (total > 0.0f) ? (1.0f / total) : 0.0f;

    /* ---- pass 3 : normalise ---- */
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        TAcc<T>::st(orw, (size_t)i, TAcc<T>::ld(orw, (size_t)i) * inv);
}

/* fused GEMM mega-kernel
 *    C = post( alpha * ( pre(A) @ B ) + bias ) + beta * C
 *  pre  : NONE | RMSNORM(gamma) | GELU | SILU | RELU   (applied to A)
 *  post : NONE | RELU | GELU | SILU | TANH | SIGMOID
 *
 *  Block layout : blockDim = (TN, TM),  one output tile of TM x TN.
 *  Shared usage : A-tile TM x (TK+1), B-tile TK x (TN+1); the RMS
 *                 reduction scratch is ALIASED onto that same storage
 *                 because its lifetime ends before the K-loop begins.*/

template <typename T, int TM, int TN, int TK>
__global__ void luaTL_gemm_fused_kernel(
        const T*     __restrict__ A,
        const T*     __restrict__ B,
        T*           __restrict__ C,
        const T*     __restrict__ bias,       /* [N]  may be NULL       */
        const T*     __restrict__ gamma,      /* [K]  may be NULL       */
        const float* __restrict__ rowscale,   /* [M]  may be NULL       */
        int M, int N, int K,
        float alpha, float beta, float eps,
        int pre_op, int post_op)
{
    const int SA    = TM * (TK + 1);
    const int SB    = TK * (TN + 1);
    const int SNEED = ((SA + SB) > (TM * TN)) ? (SA + SB) : (TM * TN);

    __shared__ float smem[SNEED];
    __shared__ float rowsc[TM];

    float* As  = smem;            /* [TM][TK+1] */
    float* Bs  = smem + SA;       /* [TK][TN+1] */
    float* red = smem;            /* aliased, used only before the K-loop */

    const int tx   = threadIdx.x;            /* 0 .. TN-1 */
    const int ty   = threadIdx.y;            /* 0 .. TM-1 */
    const int tid  = ty * TN + tx;
    const int NTHR = TM * TN;

    const int row0 = blockIdx.y * TM;
    const int col0 = blockIdx.x * TN;
    const int row  = row0 + ty;
    const int col  = col0 + tx;

    /* prologue: per-row RMS scale */
    if (pre_op == LUATL_PRE_RMSNORM) {
        if (rowscale != NULL) {
            if (tid < TM) {
                const int gr = row0 + tid;
                rowsc[tid] = (gr < M) ? rowscale[gr] : 0.0f;
            }
            __syncthreads();
        } else {
            /* TN lanes cooperate on one row; TM rows in parallel. */
            const int gr = row0 + ty;
            float part = 0.0f;
            if (gr < M) {
                for (int k = tx; k < K; k += TN) {
                    const float a = TAcc<T>::ld(A, (size_t)gr * K + k);
                    part = fmaf(a, a, part);
                }
            }
            red[ty * TN + tx] = part;
            __syncthreads();

            #pragma unroll
            for (int s = TN >> 1; s > 0; s >>= 1) {
                if (tx < s) red[ty * TN + tx] += red[ty * TN + tx + s];
                __syncthreads();
            }
            if (tx == 0)
                rowsc[ty] = rsqrtf(red[ty * TN] / (float)K + eps);
            __syncthreads();
        }
    }

    /* main K loop */
    float acc = 0.0f;

    for (int t0 = 0; t0 < K; t0 += TK) {

        /* Load the A tile, applying the prologue on the way in. */
        for (int i = tid; i < TM * TK; i += NTHR) {
            const int r  = i / TK;
            const int c  = i - r * TK;
            const int gr = row0 + r;
            const int gc = t0 + c;
            float a = (gr < M && gc < K)
                    ? TAcc<T>::ld(A, (size_t)gr * K + gc) : 0.0f;

            if (pre_op == LUATL_PRE_RMSNORM) {
                const float g = (gamma != NULL && gc < K)
                              ? TAcc<T>::ld(gamma, (size_t)gc) : 1.0f;
                a = a * rowsc[r] * g;
            } else if (pre_op != LUATL_PRE_NONE) {
                a = luatl_apply_pre(pre_op, a);
            }
            As[r * (TK + 1) + c] = a;
        }

        /* Load the B tile. */
        for (int i = tid; i < TK * TN; i += NTHR) {
            const int r  = i / TN;
            const int c  = i - r * TN;
            const int gr = t0 + r;
            const int gc = col0 + c;
            Bs[r * (TN + 1) + c] = (gr < K && gc < N)
                                 ? TAcc<T>::ld(B, (size_t)gr * N + gc) : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TK; ++k)
            acc = fmaf(As[ty * (TK + 1) + k], Bs[k * (TN + 1) + tx], acc);

        __syncthreads();
    }

    /* epilogue */
    if (row < M && col < N) {
        const size_t oi = (size_t)row * N + col;
        float v = alpha * acc;
        if (bias != NULL) v += TAcc<T>::ld(bias, (size_t)col);
        v = luatl_apply_act(post_op, v);
        if (beta != 0.0f) v += beta * TAcc<T>::ld(C, oi);
        TAcc<T>::st(C, oi, v);
    }
}

/* Tensor Core (WMMA) fp16 GEMM.  One warp per 16x16 output tile.
 *  Requires sm_70+, M/N/K all multiples of 16 and pre_op == NONE.*/
__global__ void luaTL_gemm_wmma_kernel(const __half* __restrict__ A,
                                       const __half* __restrict__ B,
                                       __half*       __restrict__ C,
                                       const __half* __restrict__ bias,
                                       int M, int N, int K,
                                       float alpha, float beta, int post_op)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    namespace wmma = nvcuda::wmma;

    __shared__ float tile[16 * 16];

    const int warpM = blockIdx.y;
    const int warpN = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + (size_t)warpM * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + (size_t)k * N + warpN * 16, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(tile, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += 32) {
        const int r = i >> 4;
        const int c = i & 15;
        const int gr = warpM * 16 + r;
        const int gc = warpN * 16 + c;
        const size_t oi = (size_t)gr * N + gc;

        float v = alpha * tile[i];
        if (bias != NULL) v += __half2float(bias[gc]);
        v = luatl_apply_act(post_op, v);
        if (beta != 0.0f) v += beta * __half2float(C[oi]);
        C[oi] = __float2half(v);
    }
#else
    (void)A; (void)B; (void)C; (void)bias;
    (void)M; (void)N; (void)K; (void)alpha; (void)beta; (void)post_op;
#endif
}

/*   Fused AdamW optimiser step
 *   fp32 master weights + optional low precision mirror copy.*/
template <typename TG>
__global__ void luaTL_adamw_kernel(float*    __restrict__ w,
                                   const TG* __restrict__ g,
                                   float*    __restrict__ m,
                                   float*    __restrict__ v,
                                   TG*       __restrict__ w_lp,
                                   uint64_t n,
                                   float lr, float beta1, float beta2,
                                   float eps, float wd,
                                   float bc1, float bc2, float clip)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float gr = TAcc<TG>::ld(g, (size_t)idx);
        if (clip > 0.0f) gr = fminf(fmaxf(gr, -clip), clip);

        const float mi = beta1 * m[idx] + (1.0f - beta1) * gr;
        const float vi = beta2 * v[idx] + (1.0f - beta2) * gr * gr;
        m[idx] = mi;
        v[idx] = vi;

        const float mh = mi * bc1;
        const float vh = vi * bc2;

        float wi = w[idx];
        wi -= lr * wd * wi;                       /* decoupled decay */
        wi -= lr * mh / (sqrtf(vh) + eps);
        w[idx] = wi;

        if (w_lp != NULL) TAcc<TG>::st(w_lp, (size_t)idx, wi);
    }
}

/*Hardware aware autotuner*/

typedef struct {
    int32_t M, N, K, dtype;
    int32_t kernel_id;
    int32_t valid;
    double  ms;
} luaTL_tune_entry_t;

static luaTL_tune_entry_t g_tune[LUATL_AUTOTUNE_SLOTS];

static int luaTL_tune_slot(int M, int N, int K, int dtype)
{
    uint64_t h = luaTL_hash_ptr((uintptr_t)((uint64_t)M * 0x9E3779B1ull ^
                                            (uint64_t)N * 0x85EBCA77ull ^
                                            (uint64_t)K * 0xC2B2AE3Dull ^
                                            (uint64_t)dtype * 0x27D4EB2Full));
    return (int)(h % (uint64_t)LUATL_AUTOTUNE_SLOTS);
}

static int luaTL_tune_lookup(int M, int N, int K, int dtype, double* ms)
{
    const int s = luaTL_tune_slot(M, N, K, dtype);
    const luaTL_tune_entry_t* e = &g_tune[s];
    if (e->valid && e->M == M && e->N == N && e->K == K && e->dtype == dtype) {
        if (ms) *ms = e->ms;
        return e->kernel_id;
    }
    return -1;
}

static void luaTL_tune_store(int M, int N, int K, int dtype,
                             int kernel_id, double ms)
{
    const int s = luaTL_tune_slot(M, N, K, dtype);
    g_tune[s].M         = M;
    g_tune[s].N         = N;
    g_tune[s].K         = K;
    g_tune[s].dtype     = dtype;
    g_tune[s].kernel_id = kernel_id;
    g_tune[s].ms        = ms;
    g_tune[s].valid     = 1;
}

/* Shared memory footprint of a (TM,TN,TK) tile in bytes. */
static int luaTL_tile_shmem(int TM, int TN, int TK)
{
    const int sa = TM * (TK + 1);
    const int sb = TK * (TN + 1);
    int need = sa + sb;
    if (TM * TN > need) need = TM * TN;
    return need * (int)sizeof(float) + TM * (int)sizeof(float);
}

static void luaTL_plan_from_id(int id, int M, int N, int K, luaTL_plan_t* p)
{
    switch (id) {
        case 1: p->tm = 32; p->tn = 32; p->tk = 32; break;   /* big square */
        case 2: p->tm =  8; p->tn = 32; p->tk = 32; break;   /* skinny M   */
        case 3: p->tm = 16; p->tn = 16; p->tk = 16; break;   /* WMMA       */
        default:
        case 0: p->tm = 16; p->tn = 16; p->tk = 16; break;   /* balanced   */
    }
    p->kernel_id       = id;
    p->block_x         = (id == 3) ? 32 : p->tn;
    p->block_y         = (id == 3) ?  1 : p->tm;
    p->grid_x          = (N + p->tn - 1) / p->tn;
    p->grid_y          = (M + p->tm - 1) / p->tm;
    p->shmem_bytes     = (id == 3) ? (16 * 16 * (int)sizeof(float))
                                   : luaTL_tile_shmem(p->tm, p->tn, p->tk);
    p->unroll          = p->tk;
    p->use_tensor_core = (id == 3) ? 1 : 0;
    p->from_cache      = 0;
    p->measured_ms     = -1.0;
    (void)K;
}

/* The heuristic core of the autotuner: pick a tile from device limits
 * and problem geometry, then validate it against the real hardware. */
static void luaTL_choose_gemm(int M, int N, int K, int dtype,
                              int allow_wmma, luaTL_plan_t* p)
{
    memset(p, 0, sizeof(*p));

    if (g_tile_override >= 0) {
        luaTL_plan_from_id(g_tile_override, M, N, K, p);
        return;
    }

    double cached_ms = -1.0;
    const int hit = luaTL_tune_lookup(M, N, K, dtype, &cached_ms);
    if (hit >= 0) {
        luaTL_plan_from_id(hit, M, N, K, p);
        p->from_cache  = 1;
        p->measured_ms = cached_ms;
        return;
    }

    int id;
    const int tc_ok = allow_wmma && (dtype == LUATL_F16) &&
                      g_devinfo.has_tensor_cores &&
                      (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0) &&
                      (M >= 32) && (N >= 32) && (K >= 32);

    if (tc_ok) {
        id = 3;
    } else if (M <= 8) {
        id = 2;                       /* decode step: 1..8 tokens        */
    } else if (M >= 128 && N >= 128 && K >= 128) {
        id = 1;                       /* fat training GEMM               */
    } else {
        id = 0;                       /* balanced default                */
    }

    luaTL_plan_from_id(id, M, N, K, p);

    /* Hardware validation: demote until the tile really fits. */
    for (int guard = 0; guard < 4; ++guard) {
        const int threads = p->block_x * p->block_y;
        const int fits_thr = (threads <= g_prop.maxThreadsPerBlock);
        const int fits_shm = ((uint64_t)p->shmem_bytes <=
                              (uint64_t)g_prop.sharedMemPerBlock);
        if (fits_thr && fits_shm) break;
        if (p->kernel_id == 1)      luaTL_plan_from_id(0, M, N, K, p);
        else if (p->kernel_id == 2) luaTL_plan_from_id(0, M, N, K, p);
        else                        break;
    }
}

/*Typed launchers (stream aware, non synchronizing)*/

template <typename T>
static int luaTL_gemm_launch(const T* A, const T* B, T* C,
                             const T* bias, const T* gamma,
                             const float* rowscale,
                             int M, int N, int K,
                             float alpha, float beta, float eps,
                             int pre_op, int post_op,
                             cudaStream_t stream)
{
    luaTL_plan_t p;
    const int allow_wmma = (pre_op == LUATL_PRE_NONE) &&
                           (TAcc<T>::dtype() == LUATL_F16);
    luaTL_choose_gemm(M, N, K, TAcc<T>::dtype(), allow_wmma, &p);

    switch (p.kernel_id) {
        case 3: {
            /* Tensor Core path is fp16-only; the template guarantees it. */
            if (TAcc<T>::dtype() == LUATL_F16) {
                dim3 blk(32, 1, 1);
                dim3 grd((unsigned)(N / 16), (unsigned)(M / 16), 1);
                luaTL_gemm_wmma_kernel<<<grd, blk, 0, stream>>>(
                    (const __half*)A, (const __half*)B, (__half*)C,
                    (const __half*)bias, M, N, K, alpha, beta, post_op);
                break;
            }
            /* fall through if somehow mis-selected */
            luaTL_plan_from_id(0, M, N, K, &p);
        }
        /* fallthrough */
        case 1: {
            if (p.kernel_id == 1) {
                dim3 blk(32, 32, 1);
                dim3 grd((unsigned)((N + 31) / 32), (unsigned)((M + 31) / 32), 1);
                luaTL_gemm_fused_kernel<T, 32, 32, 32><<<grd, blk, 0, stream>>>(
                    A, B, C, bias, gamma, rowscale,
                    M, N, K, alpha, beta, eps, pre_op, post_op);
                break;
            }
        }
        /* fallthrough */
        case 2: {
            if (p.kernel_id == 2) {
                dim3 blk(32, 8, 1);
                dim3 grd((unsigned)((N + 31) / 32), (unsigned)((M + 7) / 8), 1);
                luaTL_gemm_fused_kernel<T, 8, 32, 32><<<grd, blk, 0, stream>>>(
                    A, B, C, bias, gamma, rowscale,
                    M, N, K, alpha, beta, eps, pre_op, post_op);
                break;
            }
        }
        /* fallthrough */
        default: {
            dim3 blk(16, 16, 1);
            dim3 grd((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16), 1);
            luaTL_gemm_fused_kernel<T, 16, 16, 16><<<grd, blk, 0, stream>>>(
                A, B, C, bias, gamma, rowscale,
                M, N, K, alpha, beta, eps, pre_op, post_op);
            break;
        }
    }
    return luaTL_launch_async("gemm");
}

template <typename T>
static int luaTL_ew_launch(const T* A, const T* B, T* C, uint64_t n,
                           int op, float alpha, float beta,
                           cudaStream_t stream)
{
    const int vectorizable =
        (TAcc<T>::dtype() == LUATL_F16) && (n % 2ull == 0ull) &&
        (((uintptr_t)A & 3u) == 0u) && (((uintptr_t)C & 3u) == 0u) &&
        (B == NULL || ((uintptr_t)B & 3u) == 0u) &&
        (op == LUATL_EW_ADD || op == LUATL_EW_SUB ||
         op == LUATL_EW_MUL || op == LUATL_EW_AXPY ||
         op == LUATL_EW_SCALE) &&
        (B != NULL || op == LUATL_EW_SCALE);

    if (vectorizable) {
        const uint64_t n2 = n / 2ull;
        luaTL_ew_half2_kernel<<<luaTL_grid1d(n2, LUATL_BLOCK_1D),
                                LUATL_BLOCK_1D, 0, stream>>>(
            (const __half2*)A, (const __half2*)B, (__half2*)C,
            n2, op, alpha, beta);
    } else {
        luaTL_ew_kernel<T><<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                             LUATL_BLOCK_1D, 0, stream>>>(
            A, B, C, n, op, alpha, beta);
    }
    return luaTL_launch_async("elementwise");
}

template <typename T>
static int luaTL_rmsnorm_launch(const T* x, const T* w, T* out,
                                int rows, int cols, float eps,
                                cudaStream_t stream)
{
    const int thr = luaTL_row_threads(cols);
    luaTL_rmsnorm_kernel<T><<<(unsigned)rows, thr, 0, stream>>>(
        x, w, out, rows, cols, eps);
    return luaTL_launch_async("rmsnorm");
}

template <typename T>
static int luaTL_rms_scale_launch(const T* x, float* scale,
                                  int rows, int cols, float eps,
                                  cudaStream_t stream)
{
    const int thr = luaTL_row_threads(cols);
    luaTL_rms_scale_kernel<T><<<(unsigned)rows, thr, 0, stream>>>(
        x, scale, rows, cols, eps);
    return luaTL_launch_async("rms_scale");
}

template <typename T>
static int luaTL_softmax_launch(const T* x, const T* bias, T* out,
                                int rows, int cols, float scale,
                                int causal, int mask_offset,
                                cudaStream_t stream)
{
    const int thr = luaTL_row_threads(cols);
    luaTL_softmax_kernel<T><<<(unsigned)rows, thr, 0, stream>>>(
        x, bias, out, rows, cols, scale, causal, mask_offset);
    return luaTL_launch_async("softmax");
}

template <typename T>
static int luaTL_bcast_add_launch(const T* x, const T* v, T* out,
                                  int rows, int cols, float alpha,
                                  cudaStream_t stream)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)cols;
    luaTL_bcast_add_kernel<T><<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                                LUATL_BLOCK_1D, 0, stream>>>(
        x, v, out, rows, cols, alpha);
    return luaTL_launch_async("bcast_add");
}

template <typename T>
static int luaTL_transpose_launch(const T* in, T* out, int rows, int cols,
                                  cudaStream_t stream)
{
    dim3 blk(32, 8, 1);
    dim3 grd((unsigned)((cols + 31) / 32), (unsigned)((rows + 31) / 32), 1);
    luaTL_transpose_kernel<T><<<grd, blk, 0, stream>>>(in, out, rows, cols);
    return luaTL_launch_async("transpose");
}

/* Dispatch a templated launcher over the runtime dtype. */
#define LUATL_DISPATCH(dtype, FN, ...)                                     \
    ( (dtype) == LUATL_F16  ? FN<__half>(__VA_ARGS__)                      \
    : (dtype) == LUATL_BF16 ? FN<luatl_bf16>(__VA_ARGS__)                  \
                            : FN<float>(__VA_ARGS__) )

#define LUATL_PTR(dtype, T, p) ((T*)(p))

/*Exported C API*/

extern "C" {

/* runtime */

LUATL_API const char* luaTL_version(void) { return LUATL_VERSION_STRING; }

LUATL_API void luaTL_set_verbose(int on) { g_verbose = on ? 1 : 0; }

LUATL_API int luaTL_init(int device)
{
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) { luaTL_cuda_fail("cudaGetDeviceCount", e); return 0; }
    if (count <= 0) { luaTL_seterr("no CUDA capable device found"); return 0; }
    if (device < 0 || device >= count) device = 0;

    e = cudaSetDevice(device);
    if (e != cudaSuccess) { luaTL_cuda_fail("cudaSetDevice", e); return 0; }
    e = cudaFree(0);                                  /* force ctx create */
    if (e != cudaSuccess) { luaTL_cuda_fail("cudaFree(0)", e); return 0; }

    e = cudaGetDeviceProperties(&g_prop, device);
    if (e != cudaSuccess) { luaTL_cuda_fail("cudaGetDeviceProperties", e); return 0; }

    g_device = device;
    luaTL_fill_devinfo(device, &g_prop, &g_devinfo);

    snprintf(g_devname, sizeof(g_devname),
             "%s (sm_%d%d, %d SMs, %llu MB, %.0f GB/s, %.1f TFLOP/s fp32)",
             g_prop.name, g_prop.major, g_prop.minor,
             g_prop.multiProcessorCount,
             (unsigned long long)(g_devinfo.total_global_mem >> 20),
             g_devinfo.peak_mem_bandwidth_gbs,
             g_devinfo.peak_fp32_gflops / 1000.0);

    if (!g_pool) {
        g_pool = luaTL_pool_new(device);
        if (!g_pool) { luaTL_seterr("failed to create the default pool"); return 0; }
    }

    memset(g_tune, 0, sizeof(g_tune));
    g_initialized = 1;
    g_has_err     = 0;
    g_err[0]      = '\0';
    return count;
}

LUATL_API void luaTL_shutdown(void)
{
    if (g_pool) {
        for (int i = 0; i < g_pool->nblk; ++i) {
            if (g_pool->blks[i].alive && g_pool->blks[i].ptr) {
                cudaFree(g_pool->blks[i].ptr);
                g_pool->blks[i].ptr   = NULL;
                g_pool->blks[i].alive = 0;
            }
        }
        free(g_pool->blks);
        free(g_pool->hkey);
        free(g_pool->hval);
        free(g_pool);
        g_pool = NULL;
    }
    if (g_stage) { free(g_stage); g_stage = NULL; g_stage_size = 0; }
    cudaGetLastError();
    g_initialized = 0;
}

LUATL_API const char* luaTL_device_name(void)    { return g_devname; }
LUATL_API const char* luaTL_get_last_error(void) { return g_err; }
LUATL_API int         luaTL_has_error(void)      { return g_has_err; }

LUATL_API void luaTL_clear_error(void)
{
    g_err[0]  = '\0';
    g_has_err = 0;
    cudaGetLastError();
}

LUATL_API int luaTL_sync(void)
{
    LUATL_CHECK("cudaDeviceSynchronize", cudaDeviceSynchronize());
    return LUATL_OK;
}

LUATL_API int luaTL_get_devinfo(int device, luaTL_devinfo_t* out)
{
    if (!out) return LUATL_ERR_NULL;
    if (device < 0) device = g_device;
    cudaDeviceProp p;
    LUATL_CHECK("cudaGetDeviceProperties", cudaGetDeviceProperties(&p, device));
    luaTL_fill_devinfo(device, &p, out);
    return LUATL_OK;
}

LUATL_API int luaTL_mem_info(uint64_t* freeb, uint64_t* totalb)
{
    size_t f = 0, t = 0;
    LUATL_CHECK("cudaMemGetInfo", cudaMemGetInfo(&f, &t));
    if (freeb)  *freeb  = (uint64_t)f;
    if (totalb) *totalb = (uint64_t)t;
    return LUATL_OK;
}

/* streams */

LUATL_API void* luaTL_stream_create(int nonblocking)
{
    cudaStream_t s = NULL;
    cudaError_t e = nonblocking
        ? cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)
        : cudaStreamCreate(&s);
    if (e != cudaSuccess) { luaTL_cuda_fail("cudaStreamCreate", e); return NULL; }
    return (void*)s;
}

LUATL_API void luaTL_stream_destroy(void* s)
{
    if (!s) return;
    cudaStreamDestroy((cudaStream_t)s);
    cudaGetLastError();
}

LUATL_API int luaTL_stream_sync(void* s)
{
    LUATL_CHECK("cudaStreamSynchronize", cudaStreamSynchronize((cudaStream_t)s));
    return LUATL_OK;
}

/* pinned host memory */

LUATL_API void* luaTL_host_alloc_pinned(uint64_t bytes)
{
    void* p = NULL;
    cudaError_t e = cudaHostAlloc(&p, (size_t)bytes, cudaHostAllocPortable);
    if (e != cudaSuccess) { luaTL_cuda_fail("cudaHostAlloc", e); return NULL; }
    return p;
}

LUATL_API void luaTL_host_free_pinned(void* p)
{
    if (!p) return;
    cudaFreeHost(p);
    cudaGetLastError();
}

/* memory pool */

LUATL_API luaTL_pool_t* luaTL_pool_create(int device)
{
    if (device < 0) device = g_device;
    luaTL_pool_t* P = luaTL_pool_new(device);
    if (!P) luaTL_seterr("luaTL_pool_create: host OOM");
    return P;
}

LUATL_API void luaTL_pool_destroy(luaTL_pool_t* P)
{
    if (!P || P == g_pool) return;
    for (int i = 0; i < P->nblk; ++i) {
        if (P->blks[i].alive && P->blks[i].ptr) cudaFree(P->blks[i].ptr);
    }
    free(P->blks);
    free(P->hkey);
    free(P->hval);
    free(P);
    cudaGetLastError();
}

LUATL_API luaTL_pool_t* luaTL_default_pool(void) { return g_pool; }

LUATL_API void* luaTL_tensor_alloc(luaTL_pool_t* P, uint64_t bytes)
{
    if (!P) P = g_pool;
    if (!P) { luaTL_seterr("luaTL_tensor_alloc: no pool (call luaTL_init)"); return NULL; }
    return luaTL_pool_alloc_internal(P, bytes);
}

LUATL_API void luaTL_tensor_free(luaTL_pool_t* P, void* ptr)
{
    if (!ptr) return;
    if (!P) P = g_pool;
    luaTL_pool_free_internal(P, ptr);
}

LUATL_API void luaTL_pool_trim(luaTL_pool_t* P)
{
    if (!P) P = g_pool;
    if (!P) return;
    luaTL_pool_trim_internal(P);
}

LUATL_API int luaTL_pool_get_stats(luaTL_pool_t* P, luaTL_pool_stats_t* out)
{
    if (!out) return LUATL_ERR_NULL;
    if (!P) P = g_pool;
    if (!P) { memset(out, 0, sizeof(*out)); return LUATL_ERR_NULL; }
    *out = P->st;
    return LUATL_OK;
}

LUATL_API int luaTL_pool_owns(luaTL_pool_t* P, void* ptr)
{
    if (!P) P = g_pool;
    if (!P || !ptr) return 0;
    return (luaTL_hash_find(P, (uintptr_t)ptr) >= 0) ? 1 : 0;
}

/* device arena */

LUATL_API luaTL_arena_t* luaTL_arena_create(luaTL_pool_t* P, uint64_t bytes)
{
    if (!P) P = g_pool;
    if (!P) { luaTL_seterr("luaTL_arena_create: no pool"); return NULL; }

    luaTL_arena_t* A = (luaTL_arena_t*)calloc(1, sizeof(luaTL_arena_t));
    if (!A) { luaTL_seterr("luaTL_arena_create: host OOM"); return NULL; }

    A->base = (char*)luaTL_pool_alloc_internal(P, bytes);
    if (!A->base) { free(A); return NULL; }

    A->capacity  = bytes;
    A->offset    = 0;
    A->peak      = 0;
    A->alignment = 256;
    A->pool      = (void*)P;
    return A;
}

LUATL_API void luaTL_arena_destroy(luaTL_arena_t* A)
{
    if (!A) return;
    if (A->base) luaTL_pool_free_internal((luaTL_pool_t*)A->pool, A->base);
    free(A);
}

LUATL_API void* luaTL_arena_alloc(luaTL_arena_t* A, uint64_t bytes)
{
    if (!A || !A->base) { luaTL_seterr("arena_alloc: null arena"); return NULL; }
    const uint64_t al  = (A->alignment > 0) ? A->alignment : 256ull;
    const uint64_t off = (A->offset + al - 1ull) & ~(al - 1ull);
    if (off + bytes > A->capacity) {
        luaTL_seterr("arena_alloc: out of arena space (%llu + %llu > %llu)",
                     (unsigned long long)off, (unsigned long long)bytes,
                     (unsigned long long)A->capacity);
        return NULL;
    }
    A->offset = off + bytes;
    if (A->offset > A->peak) A->peak = A->offset;
    return (void*)(A->base + off);
}

LUATL_API void luaTL_arena_reset(luaTL_arena_t* A)
{
    if (A) A->offset = 0;
}

/* tensors */

LUATL_API int luaTL_tensor_init(luaTL_tensor_t* t, int rows, int cols,
                                int dtype, luaTL_pool_t* P)
{
    if (!t) return LUATL_ERR_NULL;
    if (rows <= 0 || cols <= 0) {
        luaTL_seterr("tensor_init: non positive shape %dx%d", rows, cols);
        return LUATL_ERR_SHAPE;
    }
    const size_t esz = luaTL_dtype_size(dtype);
    if (esz == 0) { luaTL_seterr("tensor_init: bad dtype %d", dtype); return LUATL_ERR_DTYPE; }

    if (!P) P = g_pool;
    if (!P) { luaTL_seterr("tensor_init: no pool (call luaTL_init first)"); return LUATL_ERR_NULL; }

    const uint64_t n  = (uint64_t)rows * (uint64_t)cols;
    const uint64_t nb = n * (uint64_t)esz;

    void* d = luaTL_pool_alloc_internal(P, nb);
    if (!d) return LUATL_ERR_OOM;

    t->data   = d;
    t->nelem  = n;
    t->nbytes = nb;
    t->rows   = rows;
    t->cols   = cols;
    t->dtype  = dtype;
    t->owns   = 1;
    t->pool   = (void*)P;
    t->device = g_device;
    t->flags  = 0;
    return LUATL_OK;
}

LUATL_API int luaTL_tensor_wrap(luaTL_tensor_t* t, void* data,
                                int rows, int cols, int dtype)
{
    if (!t) return LUATL_ERR_NULL;
    const size_t esz = luaTL_dtype_size(dtype);
    if (esz == 0) return LUATL_ERR_DTYPE;
    t->data   = data;
    t->nelem  = (uint64_t)rows * (uint64_t)cols;
    t->nbytes = t->nelem * (uint64_t)esz;
    t->rows   = rows;
    t->cols   = cols;
    t->dtype  = dtype;
    t->owns   = 0;
    t->pool   = NULL;
    t->device = g_device;
    t->flags  = 0;
    return LUATL_OK;
}

LUATL_API void luaTL_tensor_release(luaTL_tensor_t* t)
{
    if (!t) return;
    if (t->owns && t->data)
        luaTL_pool_free_internal((luaTL_pool_t*)t->pool, t->data);
    t->data   = NULL;
    t->owns   = 0;
    t->nelem  = 0;
    t->nbytes = 0;
}

LUATL_API int luaTL_tensor_view(luaTL_tensor_t* dst, const luaTL_tensor_t* src,
                                int row_off, int rows, int cols)
{
    if (!dst || !src || !src->data) return LUATL_ERR_NULL;
    const size_t esz = luaTL_dtype_size(src->dtype);
    if (esz == 0) return LUATL_ERR_DTYPE;
    if (cols <= 0) cols = src->cols;
    if (rows <= 0) rows = src->rows - row_off;
    if (row_off < 0 || rows <= 0) return LUATL_ERR_SHAPE;

    const uint64_t start = (uint64_t)row_off * (uint64_t)src->cols;
    const uint64_t need  = (uint64_t)rows * (uint64_t)cols;
    if (start + need > src->nelem) {
        luaTL_seterr("tensor_view: out of bounds (%llu > %llu)",
                     (unsigned long long)(start + need),
                     (unsigned long long)src->nelem);
        return LUATL_ERR_SHAPE;
    }

    dst->data   = (void*)((char*)src->data + start * esz);
    dst->nelem  = need;
    dst->nbytes = need * (uint64_t)esz;
    dst->rows   = rows;
    dst->cols   = cols;
    dst->dtype  = src->dtype;
    dst->owns   = 0;                  /* views never own storage */
    dst->pool   = NULL;
    dst->device = src->device;
    dst->flags  = src->flags;
    return LUATL_OK;
}

LUATL_API int luaTL_tensor_zero(luaTL_tensor_t* t)
{
    if (!t || !t->data) return LUATL_ERR_NULL;
    LUATL_CHECK("cudaMemset", cudaMemset(t->data, 0, (size_t)t->nbytes));
    return LUATL_OK;
}

LUATL_API int luaTL_tensor_upload(luaTL_tensor_t* t, const float* host,
                                  uint64_t nelem)
{
    if (!t || !t->data || !host) return LUATL_ERR_NULL;
    if (nelem == 0) nelem = t->nelem;
    if (nelem > t->nelem) {
        luaTL_seterr("tensor_upload: %llu elements exceeds tensor capacity %llu",
                     (unsigned long long)nelem, (unsigned long long)t->nelem);
        return LUATL_ERR_SHAPE;
    }

    if (t->dtype == LUATL_F32) {
        LUATL_CHECK("cudaMemcpy(H2D)",
                    cudaMemcpy(t->data, host, (size_t)nelem * 4,
                               cudaMemcpyHostToDevice));
        return LUATL_OK;
    }

    void* stg = luaTL_stage((size_t)nelem * 2);
    if (!stg) return LUATL_ERR_OOM;

    if (t->dtype == LUATL_F16) {
        __half* h = (__half*)stg;
        for (uint64_t i = 0; i < nelem; ++i) h[i] = __float2half(host[i]);
    } else {
        luatl_bf16* h = (luatl_bf16*)stg;
        for (uint64_t i = 0; i < nelem; ++i) h[i] = luatl_f32_to_bf16(host[i]);
    }
    LUATL_CHECK("cudaMemcpy(H2D cast)",
                cudaMemcpy(t->data, stg, (size_t)nelem * 2,
                           cudaMemcpyHostToDevice));
    return LUATL_OK;
}

LUATL_API int luaTL_tensor_download(const luaTL_tensor_t* t, float* host,
                                    uint64_t nelem)
{
    if (!t || !t->data || !host) return LUATL_ERR_NULL;
    if (nelem == 0) nelem = t->nelem;
    if (nelem > t->nelem) return LUATL_ERR_SHAPE;

    if (t->dtype == LUATL_F32) {
        LUATL_CHECK("cudaMemcpy(D2H)",
                    cudaMemcpy(host, t->data, (size_t)nelem * 4,
                               cudaMemcpyDeviceToHost));
        return LUATL_OK;
    }

    void* stg = luaTL_stage((size_t)nelem * 2);
    if (!stg) return LUATL_ERR_OOM;

    LUATL_CHECK("cudaMemcpy(D2H cast)",
                cudaMemcpy(stg, t->data, (size_t)nelem * 2,
                           cudaMemcpyDeviceToHost));

    if (t->dtype == LUATL_F16) {
        const __half* h = (const __half*)stg;
        for (uint64_t i = 0; i < nelem; ++i) host[i] = __half2float(h[i]);
    } else {
        const luatl_bf16* h = (const luatl_bf16*)stg;
        for (uint64_t i = 0; i < nelem; ++i) host[i] = luatl_bf16_to_f32(h[i]);
    }
    return LUATL_OK;
}

LUATL_API int luaTL_tensor_copy(const luaTL_tensor_t* src, luaTL_tensor_t* dst)
{
    if (!src || !dst || !src->data || !dst->data) return LUATL_ERR_NULL;
    if (src->nelem != dst->nelem) {
        luaTL_seterr("tensor_copy: element count mismatch %llu vs %llu",
                     (unsigned long long)src->nelem,
                     (unsigned long long)dst->nelem);
        return LUATL_ERR_SHAPE;
    }
    if (src->dtype == dst->dtype) {
        LUATL_CHECK("cudaMemcpy(D2D)",
                    cudaMemcpy(dst->data, src->data, (size_t)src->nbytes,
                               cudaMemcpyDeviceToDevice));
        return LUATL_OK;
    }
    return LUATL_ERR_DTYPE;   /* use luaTL_tensor_cast */
}

/* Generic device side dtype conversion. */
static int luaTL_cast_launch(const void* src, int sdt, void* dst, int ddt,
                             uint64_t n, cudaStream_t stream)
{
    const unsigned g = luaTL_grid1d(n, LUATL_BLOCK_1D);
    const int      b = LUATL_BLOCK_1D;

    if (sdt == ddt) {
        LUATL_CHECK("cudaMemcpyAsync(D2D)",
                    cudaMemcpyAsync(dst, src, (size_t)n * luaTL_dtype_size(sdt),
                                    cudaMemcpyDeviceToDevice, stream));
        return LUATL_OK;
    }

    #define CASTCASE(SD, DD, TS, TD)                                          \
        if (sdt == (SD) && ddt == (DD)) {                                     \
            luaTL_cast_kernel<TS, TD><<<g, b, 0, stream>>>(                   \
                (const TS*)src, (TD*)dst, n);                                 \
            return luaTL_launch_async("cast");                                \
        }

    CASTCASE(LUATL_F32,  LUATL_F16,  float,      __half)
    CASTCASE(LUATL_F32, LUATL_BF16, float, luatl_bf16)
    CASTCASE(LUATL_F16, LUATL_F32, __half, float)
    CASTCASE(LUATL_F16, LUATL_BF16, __half, luatl_bf16) CASTCASE(LUATL_BF16, LUATL_F32, luatl_bf16, float)
    CASTCASE(LUATL_BF16, LUATL_F16, luatl_bf16, __half)
    #undef CASTCASE

    luaTL_seterr("cast: unsupported dtype pair %d -> %d", sdt, ddt);
return LUATL_ERR_DTYPE;
}
LUATL_API int luaTL_tensor_cast(const luaTL_tensor_t* src, luaTL_tensor_t* dst) { if (!src || !dst || !src->data || !dst->data) return LUATL_ERR_NULL; if (src->nelem != dst->nelem) return LUATL_ERR_SHAPE; const int rc = luaTL_cast_launch(src->data, src->dtype, dst->data, dst->dtype, src->nelem, 0); if (rc != LUATL_OK) return rc; return luaTL_launch_sync("cast"); }

/* ------------------------- raw legacy memory API ------------------- */

LUATL_API float* luaTL_gpu_malloc(uint64_t nelem) { if (nelem == 0) { luaTL_seterr("luaTL_gpu_malloc: zero size"); return NULL; } return (float*)luaTL_tensor_alloc(g_pool, nelem * 4ull); }

LUATL_API void luaTL_gpu_free(float* p) { luaTL_tensor_free(g_pool, (void*)p); }

LUATL_API int luaTL_to_gpu(const float* host, float* dev, uint64_t nelem) { if (!host || !dev || nelem == 0) return LUATL_ERR_NULL; LUATL_CHECK("cudaMemcpy(H2D)", cudaMemcpy(dev, host, (size_t)nelem * 4, cudaMemcpyHostToDevice)); return LUATL_OK; }

LUATL_API int luaTL_to_cpu(const float* dev, float* host, uint64_t nelem) { if (!host || !dev || nelem == 0) return LUATL_ERR_NULL; LUATL_CHECK("cudaMemcpy(D2H)", cudaMemcpy(host, dev, (size_t)nelem * 4, cudaMemcpyDeviceToHost)); return LUATL_OK; }

LUATL_API int luaTL_gpu_copy(const float* src, float* dst, uint64_t nelem) { if (!src || !dst || nelem == 0) return LUATL_ERR_NULL; LUATL_CHECK("cudaMemcpy(D2D)", cudaMemcpy(dst, src, (size_t)nelem * 4, cudaMemcpyDeviceToDevice)); return LUATL_OK; }

LUATL_API int luaTL_gpu_zero(float* p, uint64_t nelem) { if (!p || nelem == 0) return LUATL_ERR_NULL; LUATL_CHECK("cudaMemset", cudaMemset(p, 0, (size_t)nelem * 4)); return LUATL_OK; }

/*autotuner API */

LUATL_API int luaTL_plan_gemm(int M, int N, int K, int dtype, int allow_tc, luaTL_plan_t* out) { if (!out) return LUATL_ERR_NULL; if (M <= 0 || N <= 0 || K <= 0) return LUATL_ERR_SHAPE; luaTL_choose_gemm(M, N, K, dtype, allow_tc, out); return LUATL_OK; }

LUATL_API void luaTL_set_tile_override(int kernel_id) { g_tile_override = kernel_id; } LUATL_API void luaTL_tune_reset(void) { memset(g_tune, 0, sizeof(g_tune)); }

/* Benchmark every legal kernel for this shape and cache the winner. */
static int luaTL_bench_one(int id, int dtype,
                           void* dA, void* dB, void* dC,
                           int M, int N, int K)
{
    g_tile_override = id;
    if (dtype == LUATL_F16)
        return luaTL_gemm_launch<__half>(
            (const __half*)dA, (const __half*)dB, (__half*)dC,
            (const __half*)NULL, (const __half*)NULL, (const float*)NULL,
            M, N, K, 1.0f, 0.0f, 1e-5f,
            LUATL_PRE_NONE, LUATL_ACT_NONE, (cudaStream_t)0);
    if (dtype == LUATL_BF16)
        return luaTL_gemm_launch<luatl_bf16>(
            (const luatl_bf16*)dA, (const luatl_bf16*)dB, (luatl_bf16*)dC,
            (const luatl_bf16*)NULL, (const luatl_bf16*)NULL, (const float*)NULL,
            M, N, K, 1.0f, 0.0f, 1e-5f,
            LUATL_PRE_NONE, LUATL_ACT_NONE, (cudaStream_t)0);
    return luaTL_gemm_launch<float>(
        (const float*)dA, (const float*)dB, (float*)dC,
        (const float*)NULL, (const float*)NULL, (const float*)NULL,
        M, N, K, 1.0f, 0.0f, 1e-5f,
        LUATL_PRE_NONE, LUATL_ACT_NONE, (cudaStream_t)0);
}

LUATL_API int luaTL_autotune_benchmark(int M, int N, int K, int dtype, int iters, luaTL_plan_t* out)
{
    if (M <= 0 || N <= 0 || K <= 0) return LUATL_ERR_SHAPE;
    if (iters <= 0) iters = 8;

    const size_t esz = luaTL_dtype_size(dtype);
    if (esz == 0) return LUATL_ERR_DTYPE;

    void *dA = NULL, *dB = NULL, *dC = NULL;
    dA = luaTL_pool_alloc_internal(g_pool, (uint64_t)M * K * esz);
    dB = luaTL_pool_alloc_internal(g_pool, (uint64_t)K * N * esz);
    dC = luaTL_pool_alloc_internal(g_pool, (uint64_t)M * N * esz);
    if (!dA || !dB || !dC) {
        luaTL_pool_free_internal(g_pool, dA);
        luaTL_pool_free_internal(g_pool, dB);
        luaTL_pool_free_internal(g_pool, dC);
        return LUATL_ERR_OOM;
    }
    cudaMemset(dA, 0, (size_t)M * K * esz);
    cudaMemset(dB, 0, (size_t)K * N * esz);
    cudaMemset(dC, 0, (size_t)M * N * esz);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    int    best_id = 0;
    double best_ms = 1e30;
    const int saved_override = g_tile_override;

    for (int id = 0; id <= 3; ++id) {
        if (id == 3) {
            if (!(dtype == LUATL_F16 && g_devinfo.has_tensor_cores &&
                  M % 16 == 0 && N % 16 == 0 && K % 16 == 0)) continue;
        }

        if (luaTL_bench_one(id, dtype, dA, dB, dC, M, N, K) != LUATL_OK) {
            cudaGetLastError();
            continue;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); continue; }

        cudaEventRecord(e0, 0);
        for (int it = 0; it < iters; ++it)
            luaTL_bench_one(id, dtype, dA, dB, dC, M, N, K);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        const double avg = (double)ms / (double)iters;
        if (cudaGetLastError() == cudaSuccess && avg < best_ms) {
            best_ms = avg;
            best_id = id;
        }
    }

    g_tile_override = saved_override;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    luaTL_pool_free_internal(g_pool, dA);
    luaTL_pool_free_internal(g_pool, dB);
    luaTL_pool_free_internal(g_pool, dC);

    if (best_ms > 1e29) { best_id = 0; best_ms = -1.0; }
    luaTL_tune_store(M, N, K, dtype, best_id, best_ms);

    if (out) {
        luaTL_plan_from_id(best_id, M, N, K, out);
        out->from_cache  = 1;
        out->measured_ms = best_ms;
    }
    return LUATL_OK;
}



/* one-shot math API  */
/* All of these synchronize; use the pipeline for latency-critical work.*/

LUATL_API int luaTL_gemm(const void* A, const void* B, void* C, const void* bias, const void* gamma, const float* rowscale, int M, int N, int K, int dtype, float alpha, float beta, float eps, int pre_op, int post_op) 
{ 
    if (!A || !B || !C) return LUATL_ERR_NULL; 
    if (M <= 0 || N <= 0 || K <= 0) return LUATL_ERR_SHAPE;

    int rc;
    if (dtype == LUATL_F16)
        rc = luaTL_gemm_launch<__half>((const __half*)A, (const __half*)B, (__half*)C, (const __half*)bias, (const __half*)gamma, rowscale, M, N, K, alpha, beta, eps, pre_op, post_op, 0);
    else if (dtype == LUATL_BF16)
        rc = luaTL_gemm_launch<luatl_bf16>((const luatl_bf16*)A, (const luatl_bf16*)B, (luatl_bf16*)C, (const luatl_bf16*)bias, (const luatl_bf16*)gamma, rowscale, M, N, K, alpha, beta, eps, pre_op, post_op, 0);
    else
        rc = luaTL_gemm_launch<float>((const float*)A, (const float*)B, (float*)C, (const float*)bias, (const float*)gamma, rowscale, M, N, K, alpha, beta, eps, pre_op, post_op, 0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("gemm");
}

LUATL_API int matmul_cuda(const float* A, const float* B, float* C, int M, int N, int K) { 
    return luaTL_gemm(A, B, C, NULL, NULL, NULL, M, N, K, LUATL_F32, 1.0f, 0.0f, 1e-5f, LUATL_PRE_NONE, LUATL_ACT_NONE); 
}

LUATL_API int fused_rmsnorm_matmul_cuda(const float* x, const float* gamma, const float* W, float* out, const float* bias, int M, int N, int K, float eps) { 
    return luaTL_gemm(x, W, out, bias, gamma, NULL, M, N, K, LUATL_F32, 1.0f, 0.0f, eps, LUATL_PRE_RMSNORM, LUATL_ACT_NONE); 
}

LUATL_API int fused_gelu_matmul_cuda(const float* x, const float* W, float* out, const float* bias, int M, int N, int K) { 
    return luaTL_gemm(x, W, out, bias, NULL, NULL, M, N, K, LUATL_F32, 1.0f, 0.0f, 1e-5f, LUATL_PRE_GELU, LUATL_ACT_NONE); 
}

LUATL_API int fused_silu_matmul_cuda(const float* x, const float* W, float* out, const float* bias, int M, int N, int K) { 
    return luaTL_gemm(x, W, out, bias, NULL, NULL, M, N, K, LUATL_F32, 1.0f, 0.0f, 1e-5f, LUATL_PRE_SILU, LUATL_ACT_NONE); 
}

LUATL_API int fused_matmul_gelu_cuda(const float* A, const float* B, float* C, const float* bias, int M, int N, int K) { 
    return luaTL_gemm(A, B, C, bias, NULL, NULL, M, N, K, LUATL_F32, 1.0f, 0.0f, 1e-5f, LUATL_PRE_NONE, LUATL_ACT_GELU); 
}

LUATL_API int luaTL_elementwise(const void* A, const void* B, void* C, uint64_t n, int dtype, int op, float alpha, float beta) { 
    if (!C || n == 0) return LUATL_ERR_NULL; 
    int rc; 
    if (dtype == LUATL_F16) 
        rc = luaTL_ew_launch<__half>((const __half*)A, (const __half*)B, (__half*)C, n, op, alpha, beta, 0); 
    else if (dtype == LUATL_BF16) 
        rc = luaTL_ew_launch<luatl_bf16>((const luatl_bf16*)A, (const luatl_bf16*)B, (luatl_bf16*)C, n, op, alpha, beta, 0); 
    else 
        rc = luaTL_ew_launch<float>((const float*)A, (const float*)B, (float*)C, n, op, alpha, beta, 0); 
    if (rc != LUATL_OK) return rc; 
    return luaTL_launch_sync("elementwise"); 
}

LUATL_API int add_cuda(const float* A, const float* B, float* C, uint64_t n) { 
    return luaTL_elementwise(A, B, C, n, LUATL_F32, LUATL_EW_ADD, 1.0f, 1.0f); 
}

LUATL_API int add_inplace_cuda(float* A, const float* B, uint64_t n, float alpha) { 
    return luaTL_elementwise(A, B, A, n, LUATL_F32, LUATL_EW_AXPY, alpha, 0.0f); 
}

LUATL_API int luaTL_rmsnorm(const void* x, const void* w, void* out, int rows, int cols, float eps, int dtype) { 
    if (!x || !out) return LUATL_ERR_NULL; 
    if (rows <= 0 || cols <= 0) return LUATL_ERR_SHAPE; 
    int rc; 
    if (dtype == LUATL_F16) 
        rc = luaTL_rmsnorm_launch<__half>((const __half*)x, (const __half*)w, (__half*)out, rows, cols, eps, 0); 
    else if (dtype == LUATL_BF16) 
        rc = luaTL_rmsnorm_launch<luatl_bf16>((const luatl_bf16*)x, (const luatl_bf16*)w, (luatl_bf16*)out, rows, cols, eps, 0); 
    else 
        rc = luaTL_rmsnorm_launch<float>((const float*)x, (const float*)w, (float*)out, rows, cols, eps, 0); 
    if (rc != LUATL_OK) return rc; 
    return luaTL_launch_sync("rmsnorm"); 
}

LUATL_API int rmsnorm_cuda(const float* x, const float* w, float* out, int rows, int cols, float eps) { 
    return luaTL_rmsnorm(x, w, out, rows, cols, eps, LUATL_F32); 
}

LUATL_API int rmsnorm_inplace_cuda(float* x, const float* w, int rows, int cols, float eps) { 
    return luaTL_rmsnorm(x, w, x, rows, cols, eps, LUATL_F32); 
}

LUATL_API int luaTL_softmax(const void* x, const void* bias, void* out, int rows, int cols, float scale, int causal, int mask_offset, int dtype) { 
    if (!x || !out) return LUATL_ERR_NULL; 
    if (rows <= 0 || cols <= 0) return LUATL_ERR_SHAPE; 
    int rc; 
    if (dtype == LUATL_F16) 
        rc = luaTL_softmax_launch<__half>((const __half*)x, (const __half*)bias, (__half*)out, rows, cols, scale, causal, mask_offset, 0); 
    else if (dtype == LUATL_BF16) 
        rc = luaTL_softmax_launch<luatl_bf16>((const luatl_bf16*)x, (const luatl_bf16*)bias, (luatl_bf16*)out, rows, cols, scale, causal, mask_offset, 0); 
    else 
        rc = luaTL_softmax_launch<float>((const float*)x, (const float*)bias, (float*)out, rows, cols, scale, causal, mask_offset, 0); 
    if (rc != LUATL_OK) return rc; 
    return luaTL_launch_sync("softmax"); 
}

LUATL_API int softmax_cuda(const float* x, float* out, int rows, int cols) { 
    return luaTL_softmax(x, NULL, out, rows, cols, 1.0f, 0, 0, LUATL_F32); 
}

LUATL_API int fused_add_bias_softmax_cuda(const float* x, const float* bias, float* out, int rows, int cols, float scale, int causal) { 
    return luaTL_softmax(x, bias, out, rows, cols, scale, causal, 0, LUATL_F32); 
}

LUATL_API int luaTL_bcast_add(const void* x, const void* v, void* out, int rows, int cols, float alpha, int dtype) { 
    if (!x || !out) return LUATL_ERR_NULL; 
    int rc; 
    if (dtype == LUATL_F16) 
        rc = luaTL_bcast_add_launch<__half>((const __half*)x, (const __half*)v, (__half*)out, rows, cols, alpha, 0); 
    else if (dtype == LUATL_BF16) 
        rc = luaTL_bcast_add_launch<luatl_bf16>((const luatl_bf16*)x, (const luatl_bf16*)v, (luatl_bf16*)out, rows, cols, alpha, 0); 
    else 
        rc = luaTL_bcast_add_launch<float>((const float*)x, (const float*)v, (float*)out, rows, cols, alpha, 0); 
    if (rc != LUATL_OK) return rc; 
    return luaTL_launch_sync("bcast_add"); 
}

LUATL_API int luaTL_transpose(const void* in, void* out, int rows, int cols, int dtype) { 
    if (!in || !out) return LUATL_ERR_NULL; 
    int rc; 
    if (dtype == LUATL_F16) 
        rc = luaTL_transpose_launch<__half>((const __half*)in, (__half*)out, rows, cols, 0); 
    else if (dtype == LUATL_BF16) 
        rc = luaTL_transpose_launch<luatl_bf16>((const luatl_bf16*)in, (luatl_bf16*)out, rows, cols, 0); 
    else 
        rc = luaTL_transpose_launch<float>((const float*)in, (float*)out, rows, cols, 0); 
    if (rc != LUATL_OK) return rc; 
    return luaTL_launch_sync("transpose"); 
}

LUATL_API int luaTL_rms_rowscale(const void* x, float* scale, int rows, int cols, float eps, int dtype) { 
    if (!x || !scale) return LUATL_ERR_NULL; 
    int rc; 
    if (dtype == LUATL_F16) 
        rc = luaTL_rms_scale_launch<__half>((const __half*)x, scale, rows, cols, eps, 0); 
    else if (dtype == LUATL_BF16) 
        rc = luaTL_rms_scale_launch<luatl_bf16>((const luatl_bf16*)x, scale, rows, cols, eps, 0); 
    else 
        rc = luaTL_rms_scale_launch<float>((const float*)x, scale, rows, cols, eps, 0); 
    if (rc != LUATL_OK) return rc; 
    return luaTL_launch_sync("rms_rowscale"); 
}

LUATL_API int luaTL_adamw(float* w, const void* g, float* m, float* v, void* w_lp, uint64_t n, int grad_dtype, float lr, float beta1, float beta2, float eps, float wd, int step, float clip) { 
    if (!w || !g || !m || !v || n == 0) return LUATL_ERR_NULL; 
    if (step < 1) step = 1;

    const float bc1 = 1.0f / (1.0f - powf(beta1, (float)step));
    const float bc2 = 1.0f / (1.0f - powf(beta2, (float)step));
    const unsigned grid = luaTL_grid1d(n, LUATL_BLOCK_1D);

    if (grad_dtype == LUATL_F16)
        luaTL_adamw_kernel<__half><<<grid, LUATL_BLOCK_1D>>>(
            w, (const __half*)g, m, v, (__half*)w_lp, n,
            lr, beta1, beta2, eps, wd, bc1, bc2, clip);
    else if (grad_dtype == LUATL_BF16)
        luaTL_adamw_kernel<luatl_bf16><<<grid, LUATL_BLOCK_1D>>>(
            w, (const luatl_bf16*)g, m, v, (luatl_bf16*)w_lp, n,
            lr, beta1, beta2, eps, wd, bc1, bc2, clip);
    else
        luaTL_adamw_kernel<float><<<grid, LUATL_BLOCK_1D>>>(
            w, (const float*)g, m, v, (float*)w_lp, n,
            lr, beta1, beta2, eps, wd, bc1, bc2, clip);

    return luaTL_launch_sync("adamw");
}

/*Batch command pipeline*/
LUATL_API luaTL_pipeline_t* luaTL_pipeline_create(int capacity, int timing) { 
    if (capacity <= 0) capacity = 512;

    luaTL_pipeline_t* P = (luaTL_pipeline_t*)calloc(1, sizeof(luaTL_pipeline_t));
    if (!P) { luaTL_seterr("pipeline_create: host OOM"); return NULL; }

    P->cmds = (luaTL_cmd_t*)calloc((size_t)capacity, sizeof(luaTL_cmd_t));
    if (!P->cmds) { free(P); luaTL_seterr("pipeline_create: host OOM"); return NULL; }
    P->capacity = capacity;

    cudaStream_t s = NULL;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        free(P->cmds); free(P);
        luaTL_seterr("pipeline_create: cudaStreamCreate failed");
        return NULL;
    }
    P->stream      = (void*)s;
    P->owns_stream = 1;
    P->timing      = timing ? 1 : 0;
    P->flags       = 1;                      /* sync at end by default */
    P->last_ms     = -1.0;

    if (P->timing) {
        cudaEvent_t a = NULL, b = NULL;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        P->ev_start = (void*)a;
        P->ev_stop  = (void*)b;
    }
    return P;
}

LUATL_API void luaTL_pipeline_destroy(luaTL_pipeline_t* P) { 
    if (!P) return; 
    if (P->ev_start) cudaEventDestroy((cudaEvent_t)P->ev_start); 
    if (P->ev_stop) cudaEventDestroy((cudaEvent_t)P->ev_stop); 
    if (P->stream && P->owns_stream) cudaStreamDestroy((cudaStream_t)P->stream); 
    free(P->cmds); 
    free(P); 
    cudaGetLastError(); 
}

LUATL_API void luaTL_pipeline_reset(luaTL_pipeline_t* P) { 
    if (P) { P->count = 0; P->status = LUATL_OK; } 
}

/* Optional host-side push helper (Lua normally writes the slots itself). */
LUATL_API int luaTL_pipeline_push(luaTL_pipeline_t* P, const luaTL_cmd_t* cmd) { 
    if (!P || !cmd) return LUATL_ERR_NULL; 
    if (P->count >= P->capacity) { 
        luaTL_seterr("pipeline_push: queue full (%d)", P->capacity); 
        return LUATL_ERR_CAPACITY; 
    } 
    P->cmds[P->count++] = *cmd; 
    return LUATL_OK; 
}

static int luaTL_exec_cmd(const luaTL_cmd_t* c, cudaStream_t s) { 
    switch (c->op) {

        case LUATL_OP_NOP:
            return LUATL_OK;

        case LUATL_OP_GEMM: {
            const int M = c->i0, N = c->i1, K = c->i2, pre = c->i3;
            if (M <= 0 || N <= 0 || K <= 0) return LUATL_ERR_SHAPE;
            if (c->dtype == LUATL_F16)
                return luaTL_gemm_launch<__half>(
                    (const __half*)c->p0, (const __half*)c->p1, (__half*)c->p2,
                    (const __half*)c->p3, (const __half*)c->p4, (const float*)c->p5,
                    M, N, K, c->f0, c->f1, c->f2, pre, c->act, s);
            if (c->dtype == LUATL_BF16)
                return luaTL_gemm_launch<luatl_bf16>(
                    (const luatl_bf16*)c->p0, (const luatl_bf16*)c->p1, (luatl_bf16*)c->p2,
                    (const luatl_bf16*)c->p3, (const luatl_bf16*)c->p4, (const float*)c->p5,
                    M, N, K, c->f0, c->f1, c->f2, pre, c->act, s);
            return luaTL_gemm_launch<float>(
                (const float*)c->p0, (const float*)c->p1, (float*)c->p2,
                (const float*)c->p3, (const float*)c->p4, (const float*)c->p5,
                M, N, K, c->f0, c->f1, c->f2, pre, c->act, s);
        }

        case LUATL_OP_EW: {
            if (c->n == 0) return LUATL_OK;
            if (c->dtype == LUATL_F16)
                return luaTL_ew_launch<__half>((const __half*)c->p0, (const __half*)c->p1,
                                               (__half*)c->p2, c->n, c->i0, c->f0, c->f1, s);
            if (c->dtype == LUATL_BF16)
                return luaTL_ew_launch<luatl_bf16>((const luatl_bf16*)c->p0, (const luatl_bf16*)c->p1,
                                                   (luatl_bf16*)c->p2, c->n, c->i0, c->f0, c->f1, s);
            return luaTL_ew_launch<float>((const float*)c->p0, (const float*)c->p1,
                                          (float*)c->p2, c->n, c->i0, c->f0, c->f1, s);
        }

        case LUATL_OP_RMSNORM: {
            const int rows = c->i0, cols = c->i1;
            if (rows <= 0 || cols <= 0) return LUATL_ERR_SHAPE;
            if (c->dtype == LUATL_F16)
                return luaTL_rmsnorm_launch<__half>((const __half*)c->p0, (const __half*)c->p1,
                                                    (__half*)c->p2, rows, cols, c->f0, s);
            if (c->dtype == LUATL_BF16)
                return luaTL_rmsnorm_launch<luatl_bf16>((const luatl_bf16*)c->p0, (const luatl_bf16*)c->p1,
                                                        (luatl_bf16*)c->p2, rows, cols, c->f0, s);
            return luaTL_rmsnorm_launch<float>((const float*)c->p0, (const float*)c->p1,
                                               (float*)c->p2, rows, cols, c->f0, s);
        }

        case LUATL_OP_SOFTMAX: {
            const int rows = c->i0, cols = c->i1, causal = c->i2, moff = c->i3;
            if (rows <= 0 || cols <= 0) return LUATL_ERR_SHAPE;
            if (c->dtype == LUATL_F16)
                return luaTL_softmax_launch<__half>((const __half*)c->p0, (const __half*)c->p1,
                                                    (__half*)c->p2, rows, cols, c->f0, causal, moff, s);
            if (c->dtype == LUATL_BF16)
                return luaTL_softmax_launch<luatl_bf16>((const luatl_bf16*)c->p0, (const luatl_bf16*)c->p1,
                                                        (luatl_bf16*)c->p2, rows, cols, c->f0, causal, moff, s);
            return luaTL_softmax_launch<float>((const float*)c->p0, (const float*)c->p1,
                                               (float*)c->p2, rows, cols, c->f0, causal, moff, s);
        }

        case LUATL_OP_CAST:
            return luaTL_cast_launch(c->p0, c->i0, c->p2, c->i1, c->n, s);

        case LUATL_OP_TRANSPOSE: {
            const int rows = c->i0, cols = c->i1;
            if (c->dtype == LUATL_F16)
                return luaTL_transpose_launch<__half>((const __half*)c->p0, (__half*)c->p2, rows, cols, s);
            if (c->dtype == LUATL_BF16)
                return luaTL_transpose_launch<luatl_bf16>((const luatl_bf16*)c->p0, (luatl_bf16*)c->p2, rows, cols, s);
            return luaTL_transpose_launch<float>((const float*)c->p0, (float*)c->p2, rows, cols, s);
        }

        case LUATL_OP_BCAST_ADD: {
            const int rows = c->i0, cols = c->i1;
            if (c->dtype == LUATL_F16)
                return luaTL_bcast_add_launch<__half>((const __half*)c->p0, (const __half*)c->p1,
                                                      (__half*)c->p2, rows, cols, c->f0, s);
            if (c->dtype == LUATL_BF16)
                return luaTL_bcast_add_launch<luatl_bf16>((const luatl_bf16*)c->p0, (const luatl_bf16*)c->p1,
                                                          (luatl_bf16*)c->p2, rows, cols, c->f0, s);
            return luaTL_bcast_add_launch<float>((const float*)c->p0, (const float*)c->p1,
                                                 (float*)c->p2, rows, cols, c->f0, s);
        }

        case LUATL_OP_ADAMW: {
            if (c->n == 0) return LUATL_OK;
            const int step = (c->i0 < 1) ? 1 : c->i0;
            const float bc1 = 1.0f / (1.0f - powf(c->f1, (float)step));
            const float bc2 = 1.0f / (1.0f - powf(c->f2, (float)step));
            const unsigned grid = luaTL_grid1d(c->n, LUATL_BLOCK_1D);
            if (c->dtype == LUATL_F16)
                luaTL_adamw_kernel<__half><<<grid, LUATL_BLOCK_1D, 0, s>>>(
                    (float*)c->p0, (const __half*)c->p1, (float*)c->p2, (float*)c->p3,
                    (__half*)c->p4, c->n, c->f0, c->f1, c->f2, c->f3, c->f4, bc1, bc2, c->f5);
            else if (c->dtype == LUATL_BF16)
                luaTL_adamw_kernel<luatl_bf16><<<grid, LUATL_BLOCK_1D, 0, s>>>(
                    (float*)c->p0, (const luatl_bf16*)c->p1, (float*)c->p2, (float*)c->p3,
                    (luatl_bf16*)c->p4, c->n, c->f0, c->f1, c->f2, c->f3, c->f4, bc1, bc2, c->f5);
            else
                luaTL_adamw_kernel<float><<<grid, LUATL_BLOCK_1D, 0, s>>>(
                    (float*)c->p0, (const float*)c->p1, (float*)c->p2, (float*)c->p3,
                    (float*)c->p4, c->n, c->f0, c->f1, c->f2, c->f3, c->f4, bc1, bc2, c->f5);
            return luaTL_launch_async("adamw");
        }

        case LUATL_OP_ZERO: {
            if (!c->p2 || c->n == 0) return LUATL_OK;
            LUATL_CHECK("cudaMemsetAsync",
                        cudaMemsetAsync(c->p2, 0, (size_t)c->n, s));
            return LUATL_OK;
        }

        case LUATL_OP_COPY: {
            if (!c->p0 || !c->p2 || c->n == 0) return LUATL_OK;
            LUATL_CHECK("cudaMemcpyAsync",
                        cudaMemcpyAsync(c->p2, c->p0, (size_t)c->n,
                                        cudaMemcpyDeviceToDevice, s));
            return LUATL_OK;
        }

        case LUATL_OP_SYNC:
            LUATL_CHECK("cudaStreamSynchronize", cudaStreamSynchronize(s));
            return LUATL_OK;

        default:
            luaTL_seterr("pipeline: unknown opcode %d", c->op);
            return LUATL_ERR_ARG;
    }
}

/* The single FFI call that runs an entire forward/backward step. */
LUATL_API int luaTL_pipeline_run(luaTL_pipeline_t* P) { 
    if (!P) return LUATL_ERR_NULL; 
    if (P->count <= 0) { P->status = LUATL_OK; return LUATL_OK; } 
    if (P->count > P->capacity) { 
        P->status = LUATL_ERR_CAPACITY; 
        luaTL_seterr("pipeline_run: count %d exceeds capacity %d", P->count, P->capacity); 
        return LUATL_ERR_CAPACITY; 
    }

    cudaStream_t s = (cudaStream_t)P->stream;

    if (P->timing && P->ev_start) cudaEventRecord((cudaEvent_t)P->ev_start, s);

    int rc = LUATL_OK;
    for (int i = 0; i < P->count; ++i) {
        rc = luaTL_exec_cmd(&P->cmds[i], s);
        if (rc != LUATL_OK) {
            luaTL_seterr("pipeline_run: command %d (op %d) failed: %s",
                         i, P->cmds[i].op, g_err);
            P->status = rc;
            cudaStreamSynchronize(s);
            cudaGetLastError();
            return rc;
        }
    }

    if (P->timing && P->ev_stop) cudaEventRecord((cudaEvent_t)P->ev_stop, s);

    if (P->flags & 1) {
        cudaError_t e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) {
            P->status = luaTL_cuda_fail("pipeline_run: stream sync", e);
            return P->status;
        }
        if (P->timing && P->ev_stop) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, (cudaEvent_t)P->ev_start,
                                     (cudaEvent_t)P->ev_stop) == cudaSuccess)
                P->last_ms = (double)ms;
        }
    }

    P->total_runs++;
    P->total_ops += (uint64_t)P->count;
    P->status = LUATL_OK;
    return LUATL_OK;
}

LUATL_API int luaTL_pipeline_wait(luaTL_pipeline_t* P) { 
    if (!P || !P->stream) return LUATL_ERR_NULL; 
    LUATL_CHECK("cudaStreamSynchronize", cudaStreamSynchronize((cudaStream_t)P->stream)); 
    if (P->timing && P->ev_stop) { 
        float ms = 0.0f; 
        if (cudaEventElapsedTime(&ms, (cudaEvent_t)P->ev_start, (cudaEvent_t)P->ev_stop) == cudaSuccess) 
            P->last_ms = (double)ms; 
    } 
    return LUATL_OK; 
}

} /* extern "C" */