/* =====================================================================
 *  luaTL_train.cu — luaTL training extension layer  (v2.1)
 *
 *  This file is a SUPERSET of luaTL_core.cu.  It textually includes
 *  luaTL_core.cu so that it shares the same translation unit (and thus
 *  sees TAcc<T>, the memory pool, the error plumbing, the launchers and
 *  the autotuner) without requiring ANY edit to that file.
 *
 *  >>> BUILD THIS FILE INSTEAD OF luaTL_core.cu. <<<
 *  The resulting library exports every old symbol plus the new ones.
 *
 *  Linux:
 *    nvcc -O3 -std=c++14 --shared -Xcompiler -fPIC \
 *         -gencode arch=compute_70,code=sm_70 \
 *         -gencode arch=compute_75,code=sm_75 \
 *         -gencode arch=compute_86,code=sm_86 \
 *         -gencode arch=compute_89,code=sm_89 \
 *         --use_fast_math -lineinfo \
 *         luaTL_train.cu -o luaTL.so
 *
 *  Windows:
 *    nvcc -O3 -std=c++14 --shared -gencode arch=compute_86,code=sm_86 ^
 *         --use_fast_math luaTL_train.cu -o luaTL.dll
 *
 *  What this adds
 *  --------------
 *   1. luaTL_gemm_ex  : fully strided + batched + transposable GEMM with
 *                       the same fused pre/post epilogues.  One kernel
 *                       covers forward, dX, dW, multi-head QK^T and PV.
 *   2. A register-tiled 64x64x16 GEMM (4x4 per thread) for big shapes.
 *   3. Backward kernels: rmsnorm_bwd, softmax_bwd, swiglu_bwd, rope
 *                        (inverse), embed_bwd, fused cross-entropy.
 *   4. Reductions: column sum, scalar sum, global L2 + on-device clip,
 *                  row argmax, row gather.
 *   5. luaTL_prog_* : a wider command buffer than luaTL_pipeline_t, able
 *                     to express a whole fwd+bwd+AdamW step, plus CUDA
 *                     Graph capture/replay for near-zero launch overhead.
 * ================================================================== */

#include <stdarg.h>          /* luaTL_core.cu uses va_list without this */
#include "luaTL_core.cu"

#include <float.h>

#define LUATL_TRAIN_VERSION_STRING "luaTL-train 2.1.0"

/* =====================================================================
 *  SECTION T0 :: shutdown-safety epoch
 *
 *  luaTL_shutdown() frees every pool block and the pool struct itself,
 *  but live Lua tensors still hold t->pool.  Their __gc then runs on
 *  freed host memory.  We bump an epoch on shutdown and make the new
 *  release path a no-op afterwards.
 * ================================================================== */
static uint64_t g_epoch = 1;

extern "C" {
LUATL_API uint64_t luaTL_epoch(void) { return g_epoch; }

LUATL_API void luaTL_shutdown_safe(void)
{
    g_epoch++;
    luaTL_shutdown();
}

/* Epoch-guarded tensor release.  Lua's __gc uses this instead of
 * luaTL_tensor_release so that post-shutdown finalizers cannot crash. */
LUATL_API void luaTL_tensor_release_safe(luaTL_tensor_t* t)
{
    if (!t) return;
    if (!g_initialized || !g_pool) { t->data = NULL; t->owns = 0; return; }
    luaTL_tensor_release(t);
}
} /* extern "C" */

/* =====================================================================
 *  SECTION T1 :: New public enums / structs
 * ================================================================== */

enum luaTL_pop_e {
    LTL_P_NOP          = 0,
    LTL_P_GEMM         = 1,
    LTL_P_EW           = 2,
    LTL_P_RMSNORM      = 3,
    LTL_P_RMSNORM_BWD  = 4,
    LTL_P_SOFTMAX      = 5,
    LTL_P_SOFTMAX_BWD  = 6,
    LTL_P_SWIGLU       = 7,
    LTL_P_SWIGLU_BWD   = 8,
    LTL_P_ROPE         = 9,
    LTL_P_EMBED        = 10,
    LTL_P_EMBED_BWD    = 11,
    LTL_P_CE           = 12,
    LTL_P_REDUCE_COLS  = 13,
    LTL_P_L2           = 14,
    LTL_P_CLIP         = 15,
    LTL_P_ADAMW        = 16,
    LTL_P_ZERO         = 17,
    LTL_P_COPY         = 18,
    LTL_P_CAST         = 19,
    LTL_P_TRANSPOSE    = 20,
    LTL_P_BCAST_ADD    = 21,
    LTL_P_SYNC         = 22,
    LTL_P_ARGMAX       = 23,
    LTL_P_SUM          = 24,
    LTL_P_GATHER       = 25,
    LTL_P_ROWRMS       = 26,
    LTL_P__COUNT       = 27
};

/* Wide command word.  Byte layout is mirrored exactly in luaTL_train.lua.
 *   12 * int32 = 48
 *    8 * float = 32   (running total 80, already 8-aligned)
 *    9 * int64 = 72   (152)
 *    1 * u64   =  8   (160)
 *    8 * void* = 64   (224)                                            */
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

/* =====================================================================
 *  SECTION T2 :: Generalised strided GEMM
 *
 *   C[b][m][n] = post( alpha * sum_k pre(A)[b][m][k] * B[b][k][n]
 *                      + bias[n] ) + beta * C[b][m][n]
 *
 *  Addressing is fully explicit:
 *     A element (m,k) lives at  A + b*a_bs + m*a_rs + k*a_cs
 *     B element (k,n) lives at  B + b*b_bs + k*b_rs + n*b_cs
 *     C element (m,n) lives at  C + b*c_bs + m*c_rs + n*c_cs
 *
 *  Therefore:
 *     row-major [M,K], ld=K       -> a_rs=ld, a_cs=1
 *     TRANSPOSED (stored [K,M])   -> a_rs=1,  a_cs=ld
 *     head h of a [T, H*hd] tensor-> a_rs=H*hd, a_cs=1, a_bs=hd
 *
 *  NOTE (inherited semantics): when post_op != NONE the activation is
 *  applied BEFORE the beta*C accumulate.  Always use act=NONE with beta!=0.
 * ================================================================== */

/* ---- shared-memory tiled variant (small / odd shapes, inline RMS) --- */
template <typename T, int TM, int TN, int TK>
__global__ void luaTL_gemm_ex_kernel(
        const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
        const T* __restrict__ bias, const T* __restrict__ gamma,
        const float* __restrict__ rowscale,
        int M, int N, int K,
        long long a_rs, long long a_cs,
        long long b_rs, long long b_cs,
        long long c_rs, long long c_cs,
        long long a_bs, long long b_bs, long long c_bs,
        float alpha, float beta, float eps, int pre_op, int post_op)
{
    __shared__ float As[TM * (TK + 1)];
    __shared__ float Bs[TK * (TN + 1)];
    __shared__ float red[TM * TN];
    __shared__ float rowsc[TM];

    const long long bz = (long long)blockIdx.z;
    A = A + bz * a_bs;
    B = B + bz * b_bs;
    C = C + bz * c_bs;

    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int tid  = ty * TN + tx;
    const int NTHR = TM * TN;

    const int row0 = blockIdx.y * TM;
    const int col0 = blockIdx.x * TN;
    const int row  = row0 + ty;
    const int col  = col0 + tx;

    /* ---- prologue: per-row inverse RMS ---- */
    if (pre_op == LUATL_PRE_RMSNORM) {
        if (rowscale != NULL) {
            if (tid < TM) {
                const int gr = row0 + tid;
                rowsc[tid] = (gr < M) ? rowscale[gr] : 0.0f;
            }
            __syncthreads();
        } else {
            const int gr = row0 + ty;
            float part = 0.0f;
            if (gr < M) {
                for (int k = tx; k < K; k += TN) {
                    const float a = TAcc<T>::ld(A, (size_t)((long long)gr * a_rs +
                                                            (long long)k  * a_cs));
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
            if (tx == 0) rowsc[ty] = rsqrtf(red[ty * TN] / (float)K + eps);
            __syncthreads();
        }
    }

    float acc = 0.0f;

    for (int t0 = 0; t0 < K; t0 += TK) {
        for (int i = tid; i < TM * TK; i += NTHR) {
            const int r  = i / TK;
            const int c  = i - r * TK;
            const int gr = row0 + r;
            const int gc = t0 + c;
            float a = (gr < M && gc < K)
                ? TAcc<T>::ld(A, (size_t)((long long)gr * a_rs + (long long)gc * a_cs))
                : 0.0f;
            if (pre_op == LUATL_PRE_RMSNORM) {
                const float g = (gamma != NULL && gc < K)
                              ? TAcc<T>::ld(gamma, (size_t)gc) : 1.0f;
                a = a * rowsc[r] * g;
            } else if (pre_op != LUATL_PRE_NONE) {
                a = luatl_apply_pre(pre_op, a);
            }
            As[r * (TK + 1) + c] = a;
        }
        for (int i = tid; i < TK * TN; i += NTHR) {
            const int r  = i / TN;
            const int c  = i - r * TN;
            const int gr = t0 + r;
            const int gc = col0 + c;
            Bs[r * (TN + 1) + c] = (gr < K && gc < N)
                ? TAcc<T>::ld(B, (size_t)((long long)gr * b_rs + (long long)gc * b_cs))
                : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TK; ++k)
            acc = fmaf(As[ty * (TK + 1) + k], Bs[k * (TN + 1) + tx], acc);

        __syncthreads();
    }

    if (row < M && col < N) {
        const size_t oi = (size_t)((long long)row * c_rs + (long long)col * c_cs);
        float v = alpha * acc;
        if (bias != NULL) v += TAcc<T>::ld(bias, (size_t)col);
        v = luatl_apply_act(post_op, v);
        if (beta != 0.0f) v += beta * TAcc<T>::ld(C, oi);
        TAcc<T>::st(C, oi, v);
    }
}

/* ---- register-tiled 64x64x16, 4x4 outputs/thread, 256 threads ------- */
#define LTL_BM 64
#define LTL_BN 64
#define LTL_BK 16

template <typename T>
__global__ void luaTL_gemm_reg_kernel(
        const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
        const T* __restrict__ bias, const T* __restrict__ gamma,
        const float* __restrict__ rowscale,
        int M, int N, int K,
        long long a_rs, long long a_cs,
        long long b_rs, long long b_cs,
        long long c_rs, long long c_cs,
        long long a_bs, long long b_bs, long long c_bs,
        float alpha, float beta, int pre_op, int post_op)
{
    __shared__ float As[LTL_BK][LTL_BM];   /* transposed: [k][m] */
    __shared__ float Bs[LTL_BK][LTL_BN];
    __shared__ float rsc[LTL_BM];

    const long long bz = (long long)blockIdx.z;
    A = A + bz * a_bs;
    B = B + bz * b_bs;
    C = C + bz * c_bs;

    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;   /* 0..255 */
    const int row0 = blockIdx.y * LTL_BM;
    const int col0 = blockIdx.x * LTL_BN;

    if (pre_op == LUATL_PRE_RMSNORM) {
        for (int i = tid; i < LTL_BM; i += 256) {
            const int gr = row0 + i;
            rsc[i] = (rowscale != NULL && gr < M) ? rowscale[gr] : 1.0f;
        }
    }
    __syncthreads();

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += LTL_BK) {

        /* A tile: 64x16 = 1024 elements over 256 threads */
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int idx = tid + it * 256;
            const int m   = idx >> 4;
            const int k   = idx & 15;
            const int gm  = row0 + m;
            const int gk  = k0 + k;
            float a = (gm < M && gk < K)
                ? TAcc<T>::ld(A, (size_t)((long long)gm * a_rs + (long long)gk * a_cs))
                : 0.0f;
            if (pre_op == LUATL_PRE_RMSNORM) {
                const float g = (gamma != NULL && gk < K)
                              ? TAcc<T>::ld(gamma, (size_t)gk) : 1.0f;
                a = a * rsc[m] * g;
            } else if (pre_op != LUATL_PRE_NONE) {
                a = luatl_apply_pre(pre_op, a);
            }
            As[k][m] = a;
        }

        /* B tile: 16x64 = 1024 elements */
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int idx = tid + it * 256;
            const int k   = idx >> 6;
            const int n   = idx & 63;
            const int gk  = k0 + k;
            const int gn  = col0 + n;
            Bs[k][n] = (gk < K && gn < N)
                ? TAcc<T>::ld(B, (size_t)((long long)gk * b_rs + (long long)gn * b_cs))
                : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < LTL_BK; ++k) {
            const float4 av = *reinterpret_cast<const float4*>(&As[k][threadIdx.y * 4]);
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][threadIdx.x * 4]);
            const float a0 = av.x, a1 = av.y, a2 = av.z, a3 = av.w;
            const float b0 = bv.x, b1 = bv.y, b2 = bv.z, b3 = bv.w;
            acc[0][0] = fmaf(a0, b0, acc[0][0]); acc[0][1] = fmaf(a0, b1, acc[0][1]);
            acc[0][2] = fmaf(a0, b2, acc[0][2]); acc[0][3] = fmaf(a0, b3, acc[0][3]);
            acc[1][0] = fmaf(a1, b0, acc[1][0]); acc[1][1] = fmaf(a1, b1, acc[1][1]);
            acc[1][2] = fmaf(a1, b2, acc[1][2]); acc[1][3] = fmaf(a1, b3, acc[1][3]);
            acc[2][0] = fmaf(a2, b0, acc[2][0]); acc[2][1] = fmaf(a2, b1, acc[2][1]);
            acc[2][2] = fmaf(a2, b2, acc[2][2]); acc[2][3] = fmaf(a2, b3, acc[2][3]);
            acc[3][0] = fmaf(a3, b0, acc[3][0]); acc[3][1] = fmaf(a3, b1, acc[3][1]);
            acc[3][2] = fmaf(a3, b2, acc[3][2]); acc[3][3] = fmaf(a3, b3, acc[3][3]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gr = row0 + threadIdx.y * 4 + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gc = col0 + threadIdx.x * 4 + j;
            if (gc >= N) continue;
            const size_t oi = (size_t)((long long)gr * c_rs + (long long)gc * c_cs);
            float v = alpha * acc[i][j];
            if (bias != NULL) v += TAcc<T>::ld(bias, (size_t)gc);
            v = luatl_apply_act(post_op, v);
            if (beta != 0.0f) v += beta * TAcc<T>::ld(C, oi);
            TAcc<T>::st(C, oi, v);
        }
    }
}

/* ---- strided per-row inverse-RMS (feeds the register kernel) -------- */
template <typename T>
__global__ void luaTL_rowrms_kernel(const T* __restrict__ x,
                                    float* __restrict__ out,
                                    int rows, int cols,
                                    long long rs, long long cs, float eps)
{
    __shared__ float red[32];
    const int r = blockIdx.x;
    float local = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = TAcc<T>::ld(x, (size_t)((long long)r * rs + (long long)i * cs));
        local = fmaf(v, v, local);
    }
    const float tot = luatl_block_sum(local, red);
    if (threadIdx.x == 0) out[r] = rsqrtf(tot / (float)cols + eps);
}

/* Persistent scratch for auto-computed RMS row scales (never in the
 * pool, so it is safe to reuse inside a captured CUDA graph). */
static float*  g_rowscale     = NULL;
static int     g_rowscale_cap = 0;

static float* luaTL_rowscale_scratch(int rows)
{
    if (rows <= g_rowscale_cap && g_rowscale) return g_rowscale;
    if (g_rowscale) { cudaFree(g_rowscale); g_rowscale = NULL; g_rowscale_cap = 0; }
    if (cudaMalloc((void**)&g_rowscale, (size_t)rows * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        g_rowscale = NULL;
        return NULL;
    }
    g_rowscale_cap = rows;
    return g_rowscale;
}

/* ---- host launcher -------------------------------------------------- */
template <typename T>
static int luaTL_gemm_ex_launch(
        const void* A_, const void* B_, void* C_,
        const void* bias_, const void* gamma_, const float* rowscale,
        int M, int N, int K,
        long long a_rs, long long a_cs, long long b_rs, long long b_cs,
        long long c_rs, long long c_cs,
        long long a_bs, long long b_bs, long long c_bs,
        int batch, float alpha, float beta, float eps,
        int pre_op, int post_op, cudaStream_t s)
{
    const T*     A  = (const T*)A_;
    const T*     B  = (const T*)B_;
    T*           C  = (T*)C_;
    const T*     bs = (const T*)bias_;
    const T*     gm = (const T*)gamma_;
    if (batch < 1) batch = 1;

    /* --- Tensor Core fast path: fp16, canonical layout, single batch --- */
    if (TAcc<T>::dtype() == LUATL_F16 && batch == 1 &&
        pre_op == LUATL_PRE_NONE && g_devinfo.has_tensor_cores &&
        a_cs == 1 && b_cs == 1 && c_cs == 1 &&
        a_rs == (long long)K && b_rs == (long long)N && c_rs == (long long)N &&
        (M % 16) == 0 && (N % 16) == 0 && (K % 16) == 0 &&
        M >= 32 && N >= 32 && K >= 32 && g_tile_override < 0)
    {
        dim3 blk(32, 1, 1);
        dim3 grd((unsigned)(N / 16), (unsigned)(M / 16), 1);
        luaTL_gemm_wmma_kernel<<<grd, blk, 0, s>>>(
            (const __half*)A_, (const __half*)B_, (__half*)C_,
            (const __half*)bias_, M, N, K, alpha, beta, post_op);
        return luaTL_launch_async("gemm_ex(wmma)");
    }

    /* --- register-tiled path for the fat training GEMMs --------------- */
    const int big = (M >= 64 && N >= 64 && K >= 8) && (g_tile_override < 0);
    if (big) {
        const float* rsc = rowscale;
        if (pre_op == LUATL_PRE_RMSNORM && rsc == NULL) {
            float* tmp = luaTL_rowscale_scratch(M);
            if (tmp) {
                const int thr = luaTL_row_threads(K);
                luaTL_rowrms_kernel<T><<<(unsigned)M, thr, 0, s>>>(
                    A, tmp, M, K, a_rs, a_cs, eps);
                int rc = luaTL_launch_async("gemm_ex(rowrms)");
                if (rc != LUATL_OK) return rc;
                rsc = tmp;
            }
        }
        if (!(pre_op == LUATL_PRE_RMSNORM && rsc == NULL)) {
            dim3 blk(16, 16, 1);
            dim3 grd((unsigned)((N + LTL_BN - 1) / LTL_BN),
                     (unsigned)((M + LTL_BM - 1) / LTL_BM),
                     (unsigned)batch);
            luaTL_gemm_reg_kernel<T><<<grd, blk, 0, s>>>(
                A, B, C, bs, gm, rsc, M, N, K,
                a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a_bs, b_bs, c_bs,
                alpha, beta, pre_op, post_op);
            return luaTL_launch_async("gemm_ex(reg)");
        }
    }

    /* --- shared-memory tiled fallback --------------------------------- */
    if (M <= 8) {
        dim3 blk(32, 8, 1);
        dim3 grd((unsigned)((N + 31) / 32), (unsigned)((M + 7) / 8), (unsigned)batch);
        luaTL_gemm_ex_kernel<T, 8, 32, 32><<<grd, blk, 0, s>>>(
            A, B, C, bs, gm, rowscale, M, N, K,
            a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a_bs, b_bs, c_bs,
            alpha, beta, eps, pre_op, post_op);
    } else if (M >= 128 && N >= 128 && K >= 32) {
        dim3 blk(32, 32, 1);
        dim3 grd((unsigned)((N + 31) / 32), (unsigned)((M + 31) / 32), (unsigned)batch);
        luaTL_gemm_ex_kernel<T, 32, 32, 32><<<grd, blk, 0, s>>>(
            A, B, C, bs, gm, rowscale, M, N, K,
            a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a_bs, b_bs, c_bs,
            alpha, beta, eps, pre_op, post_op);
    } else {
        dim3 blk(16, 16, 1);
        dim3 grd((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16), (unsigned)batch);
        luaTL_gemm_ex_kernel<T, 16, 16, 16><<<grd, blk, 0, s>>>(
            A, B, C, bs, gm, rowscale, M, N, K,
            a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a_bs, b_bs, c_bs,
            alpha, beta, eps, pre_op, post_op);
    }
    return luaTL_launch_async("gemm_ex");
}

/* Type-correct runtime dtype dispatch (all launchers below take void*). */
#define LTL_DISPATCH(dt, FN, ...)                                          \
    ( (dt) == LUATL_F16  ? FN<__half>(__VA_ARGS__)                         \
    : (dt) == LUATL_BF16 ? FN<luatl_bf16>(__VA_ARGS__)                     \
                         : FN<float>(__VA_ARGS__) )

/* =====================================================================
 *  SECTION T3 :: RMSNorm backward
 *
 *   y_i = g_i * x_i * s ,  s = rsqrt(mean(x^2) + eps)
 *   dx_i = s*g_i*dy_i - (x_i * s^3 / N) * sum_j (dy_j * g_j * x_j)
 *   dg_i = sum_over_rows( dy_i * x_i * s )      [fp32, atomic accumulate]
 * ================================================================== */
template <typename T>
__global__ void luaTL_rmsnorm_bwd_kernel(const T* __restrict__ x,
                                         const T* __restrict__ g,
                                         const T* __restrict__ dy,
                                         T*       __restrict__ dx,
                                         float*   __restrict__ dgamma,
                                         int rows, int cols, float eps)
{
    __shared__ float red[32];
    const int r = blockIdx.x;
    const T* xr  = x  + (size_t)r * cols;
    const T* dyr = dy + (size_t)r * cols;
    T*       dxr = dx + (size_t)r * cols;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = TAcc<T>::ld(xr, (size_t)i);
        ss = fmaf(v, v, ss);
    }
    ss = luatl_block_sum(ss, red);
    const float s = rsqrtf(ss / (float)cols + eps);

    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float gi = (g != NULL) ? TAcc<T>::ld(g, (size_t)i) : 1.0f;
        dot += TAcc<T>::ld(dyr, (size_t)i) * gi * TAcc<T>::ld(xr, (size_t)i);
    }
    dot = luatl_block_sum(dot, red);

    const float c = (s * s * s / (float)cols) * dot;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float xi  = TAcc<T>::ld(xr,  (size_t)i);
        const float dyi = TAcc<T>::ld(dyr, (size_t)i);
        const float gi  = (g != NULL) ? TAcc<T>::ld(g, (size_t)i) : 1.0f;
        TAcc<T>::st(dxr, (size_t)i, s * gi * dyi - c * xi);
        if (dgamma != NULL) atomicAdd(&dgamma[i], dyi * xi * s);
    }
}

template <typename T>
static int luaTL_rmsnorm_bwd_launch(const void* x, const void* g, const void* dy,
                                    void* dx, float* dgamma,
                                    int rows, int cols, float eps, cudaStream_t s)
{
    const int thr = luaTL_row_threads(cols);
    luaTL_rmsnorm_bwd_kernel<T><<<(unsigned)rows, thr, 0, s>>>(
        (const T*)x, (const T*)g, (const T*)dy, (T*)dx, dgamma, rows, cols, eps);
    return luaTL_launch_async("rmsnorm_bwd");
}

/* =====================================================================
 *  SECTION T4 :: Group-aware softmax (fixes multi-head causal masking)
 *                and its backward.
 *
 *  `group` : number of rows per logical sequence.  Row r belongs to
 *            position (r % group).  Pass group = T when scores are
 *            packed as [H*T, T]; pass 0/rows for a plain matrix.
 * ================================================================== */
template <typename T>
__global__ void luaTL_softmax_ex_kernel(const T* __restrict__ x,
                                        const T* __restrict__ bias,
                                        T*       __restrict__ out,
                                        int rows, int cols, float scale,
                                        int causal, int group, int mask_offset)
{
    __shared__ float red[32];
    const int row  = blockIdx.x;
    const int lrow = (group > 0) ? (row % group) : row;
    const T*  xr   = x   + (size_t)row * cols;
    T*        orw  = out + (size_t)row * cols;

    const int limit = causal ? (lrow + mask_offset) : (cols - 1);

    float lmax = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i > limit) continue;
        float v = TAcc<T>::ld(xr, (size_t)i) * scale;
        if (bias != NULL) v += TAcc<T>::ld(bias, (size_t)i);
        lmax = fmaxf(lmax, v);
    }
    const float rmax = luatl_block_max(lmax, red);
    const float base = (rmax == -INFINITY) ? 0.0f : rmax;

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
    const float tot = luatl_block_sum(lsum, red);
    const float inv = (tot > 0.0f) ? (1.0f / tot) : 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        TAcc<T>::st(orw, (size_t)i, TAcc<T>::ld(orw, (size_t)i) * inv);
}

/*  dx_i = scale * y_i * ( dy_i - sum_j dy_j * y_j )   (masked entries = 0) */
template <typename T>
__global__ void luaTL_softmax_bwd_kernel(const T* __restrict__ y,
                                         const T* __restrict__ dy,
                                         T*       __restrict__ dx,
                                         int rows, int cols, float scale,
                                         int causal, int group, int mask_offset)
{
    __shared__ float red[32];
    const int row  = blockIdx.x;
    const int lrow = (group > 0) ? (row % group) : row;
    const T*  yr   = y  + (size_t)row * cols;
    const T*  dyr  = dy + (size_t)row * cols;
    T*        dxr  = dx + (size_t)row * cols;

    const int limit = causal ? (lrow + mask_offset) : (cols - 1);

    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i > limit) continue;
        dot += TAcc<T>::ld(dyr, (size_t)i) * TAcc<T>::ld(yr, (size_t)i);
    }
    dot = luatl_block_sum(dot, red);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = 0.0f;
        if (i <= limit) {
            const float yi = TAcc<T>::ld(yr, (size_t)i);
            v = scale * yi * (TAcc<T>::ld(dyr, (size_t)i) - dot);
        }
        TAcc<T>::st(dxr, (size_t)i, v);
    }
}

template <typename T>
static int luaTL_softmax_ex_launch(const void* x, const void* bias, void* out,
                                   int rows, int cols, float scale,
                                   int causal, int group, int moff, cudaStream_t s)
{
    const int thr = luaTL_row_threads(cols);
    luaTL_softmax_ex_kernel<T><<<(unsigned)rows, thr, 0, s>>>(
        (const T*)x, (const T*)bias, (T*)out, rows, cols, scale, causal, group, moff);
    return luaTL_launch_async("softmax_ex");
}

template <typename T>
static int luaTL_softmax_bwd_launch(const void* y, const void* dy, void* dx,
                                    int rows, int cols, float scale,
                                    int causal, int group, int moff, cudaStream_t s)
{
    const int thr = luaTL_row_threads(cols);
    luaTL_softmax_bwd_kernel<T><<<(unsigned)rows, thr, 0, s>>>(
        (const T*)y, (const T*)dy, (T*)dx, rows, cols, scale, causal, group, moff);
    return luaTL_launch_async("softmax_bwd");
}

/* =====================================================================
 *  SECTION T5 :: SwiGLU  (out = silu(a) * b)
 *
 *  Strided so that the packed layout works too: for a single [rows,2*cols]
 *  tensor pass a = base, b = base + cols, lda = ldb = 2*cols.
 * ================================================================== */
template <typename T>
__global__ void luaTL_swiglu_fwd_kernel(const T* __restrict__ a,
                                        const T* __restrict__ b,
                                        T*       __restrict__ o,
                                        int rows, int cols,
                                        int lda, int ldb, int ldo)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)cols;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += st) {
        const int r = (int)(idx / (uint64_t)cols);
        const int c = (int)(idx - (uint64_t)r * (uint64_t)cols);
        const float av = TAcc<T>::ld(a, (size_t)r * lda + c);
        const float bv = TAcc<T>::ld(b, (size_t)r * ldb + c);
        TAcc<T>::st(o, (size_t)r * ldo + c, luatl_silu(av) * bv);
    }
}

template <typename T>
__global__ void luaTL_swiglu_bwd_kernel(const T* __restrict__ a,
                                        const T* __restrict__ b,
                                        const T* __restrict__ dout,
                                        T*       __restrict__ da,
                                        T*       __restrict__ db,
                                        int rows, int cols,
                                        int lda, int ldb, int ldo)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)cols;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += st) {
        const int r = (int)(idx / (uint64_t)cols);
        const int c = (int)(idx - (uint64_t)r * (uint64_t)cols);
        const float av = TAcc<T>::ld(a,    (size_t)r * lda + c);
        const float bv = TAcc<T>::ld(b,    (size_t)r * ldb + c);
        const float go = TAcc<T>::ld(dout, (size_t)r * ldo + c);
        if (da) TAcc<T>::st(da, (size_t)r * lda + c, go * bv * luatl_silu_grad(av));
        if (db) TAcc<T>::st(db, (size_t)r * ldb + c, go * luatl_silu(av));
    }
}

template <typename T>
static int luaTL_swiglu_fwd_launch(const void* a, const void* b, void* o,
                                   int rows, int cols, int lda, int ldb, int ldo,
                                   cudaStream_t s)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)cols;
    luaTL_swiglu_fwd_kernel<T><<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                                 LUATL_BLOCK_1D, 0, s>>>(
        (const T*)a, (const T*)b, (T*)o, rows, cols, lda, ldb, ldo);
    return luaTL_launch_async("swiglu_fwd");
}

template <typename T>
static int luaTL_swiglu_bwd_launch(const void* a, const void* b, const void* dout,
                                   void* da, void* db,
                                   int rows, int cols, int lda, int ldb, int ldo,
                                   cudaStream_t s)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)cols;
    luaTL_swiglu_bwd_kernel<T><<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                                 LUATL_BLOCK_1D, 0, s>>>(
        (const T*)a, (const T*)b, (const T*)dout, (T*)da, (T*)db,
        rows, cols, lda, ldb, ldo);
    return luaTL_launch_async("swiglu_bwd");
}

/* =====================================================================
 *  SECTION T6 :: RoPE (forward and inverse share one kernel)
 *
 *  layout 0 = half-split (Llama / GPT-NeoX):  pairs (i, i + hd/2)
 *  layout 1 = interleaved (GPT-J):            pairs (2i, 2i+1)
 *  inverse = 1 negates the sine, which is exactly the backward pass.
 * ================================================================== */
template <typename T>
__global__ void luaTL_rope_kernel(T* __restrict__ x,
                                  const int32_t* __restrict__ pos_ids,
                                  int rows, int heads, int hd, int row_stride,
                                  int pos_offset, float theta,
                                  int interleaved, int inverse)
{
    const int pairs = hd >> 1;
    const uint64_t n = (uint64_t)rows * (uint64_t)heads * (uint64_t)pairs;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += st) {
        const int p   = (int)(idx % (uint64_t)pairs);
        const int hh  = (int)((idx / (uint64_t)pairs) % (uint64_t)heads);
        const int r   = (int)(idx / ((uint64_t)pairs * (uint64_t)heads));

        const int pos = (pos_ids != NULL) ? pos_ids[r] : (r + pos_offset);
        const float freq = __powf(theta, -2.0f * (float)p / (float)hd);
        const float ang  = (float)pos * freq;
        float c, s;
        __sincosf(ang, &s, &c);
        if (inverse) s = -s;

        const size_t base = (size_t)r * row_stride + (size_t)hh * hd;
        size_t i0, i1;
        if (interleaved) { i0 = base + 2 * p; i1 = base + 2 * p + 1; }
        else             { i0 = base + p;     i1 = base + p + pairs; }

        const float x0 = TAcc<T>::ld(x, i0);
        const float x1 = TAcc<T>::ld(x, i1);
        TAcc<T>::st(x, i0, x0 * c - x1 * s);
        TAcc<T>::st(x, i1, x1 * c + x0 * s);
    }
}

template <typename T>
static int luaTL_rope_launch(void* x, const int32_t* pos_ids,
                             int rows, int heads, int hd, int row_stride,
                             int pos_offset, float theta,
                             int interleaved, int inverse, cudaStream_t s)
{
    if ((hd & 1) != 0) { luaTL_seterr("rope: head_dim must be even"); return LUATL_ERR_ARG; }
    const uint64_t n = (uint64_t)rows * (uint64_t)heads * (uint64_t)(hd >> 1);
    luaTL_rope_kernel<T><<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                           LUATL_BLOCK_1D, 0, s>>>(
        (T*)x, pos_ids, rows, heads, hd, row_stride,
        pos_offset, theta, interleaved, inverse);
    return luaTL_launch_async("rope");
}

/* =====================================================================
 *  SECTION T7 :: Embedding gather / scatter-add
 * ================================================================== */
template <typename T>
__global__ void luaTL_embed_fwd_kernel(const int32_t* __restrict__ ids,
                                       const T* __restrict__ table,
                                       T* __restrict__ out,
                                       int rows, int dim, int vocab)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)dim;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += st) {
        const int r = (int)(idx / (uint64_t)dim);
        const int c = (int)(idx - (uint64_t)r * (uint64_t)dim);
        const int t = ids[r];
        float v = 0.0f;
        if (t >= 0 && t < vocab) v = TAcc<T>::ld(table, (size_t)t * dim + c);
        TAcc<T>::st(out, (size_t)r * dim + c, v);
    }
}

template <typename T>
__global__ void luaTL_embed_bwd_kernel(const int32_t* __restrict__ ids,
                                       const T* __restrict__ dout,
                                       float* __restrict__ dtable,
                                       int rows, int dim, int vocab, float scale)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)dim;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += st) {
        const int r = (int)(idx / (uint64_t)dim);
        const int c = (int)(idx - (uint64_t)r * (uint64_t)dim);
        const int t = ids[r];
        if (t < 0 || t >= vocab) continue;
        atomicAdd(&dtable[(size_t)t * dim + c],
                  scale * TAcc<T>::ld(dout, (size_t)r * dim + c));
    }
}

template <typename T>
static int luaTL_embed_fwd_launch(const void* ids, const void* table, void* out,
                                  int rows, int dim, int vocab, cudaStream_t s)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)dim;
    luaTL_embed_fwd_kernel<T><<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                                LUATL_BLOCK_1D, 0, s>>>(
        (const int32_t*)ids, (const T*)table, (T*)out, rows, dim, vocab);
    return luaTL_launch_async("embed_fwd");
}

template <typename T>
static int luaTL_embed_bwd_launch(const void* ids, const void* dout, void* dtable,
                                  int rows, int dim, int vocab, float scale,
                                  cudaStream_t s)
{
    const uint64_t n = (uint64_t)rows * (uint64_t)dim;
    luaTL_embed_bwd_kernel<T><<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                                LUATL_BLOCK_1D, 0, s>>>(
        (const int32_t*)ids, (const T*)dout, (float*)dtable,
        rows, dim, vocab, scale);
    return luaTL_launch_async("embed_bwd");
}

/* =====================================================================
 *  SECTION T8 :: Fused softmax + cross-entropy + dlogits
 *
 *  Keeps the [rows, vocab] logits on the GPU.  Emits only a float[rows]
 *  loss vector, which is 4 bytes/token instead of 4*V bytes/token.
 * ================================================================== */
template <typename T>
__global__ void luaTL_ce_kernel(const T* __restrict__ logits,
                                const int32_t* __restrict__ targets,
                                float* __restrict__ losses,
                                T* __restrict__ dlogits,
                                int rows, int V, int ignore_index, float gscale)
{
    __shared__ float red[32];
    const int r = blockIdx.x;
    const T* lr = logits + (size_t)r * V;
    T*       dr = (dlogits != NULL) ? (dlogits + (size_t)r * V) : NULL;
    const int t = targets[r];

    if (t < 0 || t >= V || t == ignore_index) {
        if (threadIdx.x == 0 && losses) losses[r] = 0.0f;
        if (dr) for (int i = threadIdx.x; i < V; i += blockDim.x)
                    TAcc<T>::st(dr, (size_t)i, 0.0f);
        return;
    }

    float lmax = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        lmax = fmaxf(lmax, TAcc<T>::ld(lr, (size_t)i));
    const float mx = luatl_block_max(lmax, red);

    float lsum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        lsum += __expf(TAcc<T>::ld(lr, (size_t)i) - mx);
    const float sum = luatl_block_sum(lsum, red);
    const float lse = logf(fmaxf(sum, 1e-30f)) + mx;

    if (threadIdx.x == 0 && losses)
        losses[r] = lse - TAcc<T>::ld(lr, (size_t)t);

    if (dr) {
        const float inv = 1.0f / fmaxf(sum, 1e-30f);
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            float p = __expf(TAcc<T>::ld(lr, (size_t)i) - mx) * inv;
            if (i == t) p -= 1.0f;
            TAcc<T>::st(dr, (size_t)i, gscale * p);
        }
    }
}

template <typename T>
static int luaTL_ce_launch(const void* logits, const void* targets,
                           float* losses, void* dlogits,
                           int rows, int V, int ignore_index, float gscale,
                           cudaStream_t s)
{
    const int thr = luaTL_row_threads(V);
    luaTL_ce_kernel<T><<<(unsigned)rows, thr, 0, s>>>(
        (const T*)logits, (const int32_t*)targets, losses, (T*)dlogits,
        rows, V, ignore_index, gscale);
    return luaTL_launch_async("cross_entropy");
}

/* =====================================================================
 *  SECTION T9 :: Reductions, clipping, argmax, gather
 * ================================================================== */
template <typename T>
__global__ void luaTL_reduce_cols_kernel(const T* __restrict__ x,
                                         float* __restrict__ out,
                                         int rows, int cols,
                                         long long rs, float alpha, float beta)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float acc = 0.0f;
    for (int r = 0; r < rows; ++r)
        acc += TAcc<T>::ld(x, (size_t)((long long)r * rs + (long long)c));
    out[c] = alpha * acc + beta * out[c];
}

template <typename T>
static int luaTL_reduce_cols_launch(const void* x, float* out, int rows, int cols,
                                    long long rs, float alpha, float beta,
                                    cudaStream_t s)
{
    const int blk = 128;
    const unsigned grd = (unsigned)((cols + blk - 1) / blk);
    luaTL_reduce_cols_kernel<T><<<grd, blk, 0, s>>>(
        (const T*)x, out, rows, cols, rs, alpha, beta);
    return luaTL_launch_async("reduce_cols");
}

template <typename T>
__global__ void luaTL_l2sq_kernel(const T* __restrict__ x, uint64_t n,
                                  float* __restrict__ out)
{
    __shared__ float red[32];
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; idx < n; idx += st) {
        const float v = TAcc<T>::ld(x, (size_t)idx);
        acc = fmaf(v, v, acc);
    }
    acc = luatl_block_sum(acc, red);
    if (threadIdx.x == 0) atomicAdd(out, acc);
}

template <typename T>
static int luaTL_l2_launch(const void* x, uint64_t n, float* out, cudaStream_t s)
{
    luaTL_l2sq_kernel<T><<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                           LUATL_BLOCK_1D, 0, s>>>((const T*)x, n, out);
    return luaTL_launch_async("l2_accum");
}

/* Scale x in place by min(1, max_norm / sqrt(*nsq)).  The norm never
 * leaves the device, so gradient clipping costs zero synchronisation. */
template <typename T>
__global__ void luaTL_clip_kernel(T* __restrict__ x, uint64_t n,
                                  const float* __restrict__ nsq, float max_norm)
{
    const float nrm = sqrtf(fmaxf(nsq[0], 0.0f));
    float f = 1.0f;
    if (max_norm > 0.0f && nrm > max_norm) f = max_norm / (nrm + 1e-6f);
    if (f == 1.0f) return;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += st)
        TAcc<T>::st(x, (size_t)idx, TAcc<T>::ld(x, (size_t)idx) * f);
}

template <typename T>
static int luaTL_clip_launch(void* x, uint64_t n, const float* nsq,
                             float max_norm, cudaStream_t s)
{
    luaTL_clip_kernel<T><<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                           LUATL_BLOCK_1D, 0, s>>>((T*)x, n, nsq, max_norm);
    return luaTL_launch_async("clip_scale");
}

__global__ void luaTL_sum_f32_kernel(const float* __restrict__ x, uint64_t n,
                                     float* __restrict__ out, float scale)
{
    __shared__ float red[32];
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; idx < n; idx += st) acc += x[idx];
    acc = luatl_block_sum(acc, red);
    if (threadIdx.x == 0) atomicAdd(out, acc * scale);
}

template <typename T>
__global__ void luaTL_argmax_kernel(const T* __restrict__ x, int32_t* __restrict__ out,
                                    int rows, int cols)
{
    __shared__ float vred[32];
    __shared__ int   ired[32];
    const int row = blockIdx.x;
    const T*  xr  = x + (size_t)row * cols;

    float best = -INFINITY;
    int   bi   = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = TAcc<T>::ld(xr, (size_t)i);
        if (v > best) { best = v; bi = i; }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, best, o);
        const int   oi = __shfl_down_sync(0xFFFFFFFFu, bi,   o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nw   = (blockDim.x + 31) >> 5;
    if (lane == 0) { vred[wid] = best; ired[wid] = bi; }
    __syncthreads();
    if (wid == 0) {
        float v = (lane < nw) ? vred[lane] : -INFINITY;
        int   i = (lane < nw) ? ired[lane] : 0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, v, o);
            const int   oi = __shfl_down_sync(0xFFFFFFFFu, i, o);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) out[row] = i;
    }
}

template <typename T>
static int luaTL_argmax_launch(const void* x, void* out, int rows, int cols,
                               cudaStream_t s)
{
    const int thr = luaTL_row_threads(cols);
    luaTL_argmax_kernel<T><<<(unsigned)rows, thr, 0, s>>>(
        (const T*)x, (int32_t*)out, rows, cols);
    return luaTL_launch_async("argmax");
}

template <typename T>
__global__ void luaTL_gather_rows_kernel(const T* __restrict__ src,
                                         const int32_t* __restrict__ idx,
                                         T* __restrict__ dst,
                                         int n, int cols, int src_rows)
{
    const uint64_t total = (uint64_t)n * (uint64_t)cols;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    for (; i < total; i += st) {
        const int r = (int)(i / (uint64_t)cols);
        const int c = (int)(i - (uint64_t)r * (uint64_t)cols);
        const int sr = idx[r];
        float v = 0.0f;
        if (sr >= 0 && sr < src_rows) v = TAcc<T>::ld(src, (size_t)sr * cols + c);
        TAcc<T>::st(dst, (size_t)r * cols + c, v);
    }
}

template <typename T>
static int luaTL_gather_launch(const void* src, const void* idx, void* dst,
                               int n, int cols, int src_rows, cudaStream_t s)
{
    const uint64_t total = (uint64_t)n * (uint64_t)cols;
    luaTL_gather_rows_kernel<T><<<luaTL_grid1d(total, LUATL_BLOCK_1D),
                                  LUATL_BLOCK_1D, 0, s>>>(
        (const T*)src, (const int32_t*)idx, (T*)dst, n, cols, src_rows);
    return luaTL_launch_async("gather_rows");
}

template <typename T>
static int luaTL_rowrms_launch(const void* x, float* out, int rows, int cols,
                               long long rs, long long cs, float eps, cudaStream_t s)
{
    const int thr = luaTL_row_threads(cols);
    luaTL_rowrms_kernel<T><<<(unsigned)rows, thr, 0, s>>>(
        (const T*)x, out, rows, cols, rs, cs, eps);
    return luaTL_launch_async("rowrms");
}

/* =====================================================================
 *  SECTION T10 :: Exported C API for everything above
 * ================================================================== */
extern "C" {

LUATL_API const char* luaTL_train_version(void) { return LUATL_TRAIN_VERSION_STRING; }

LUATL_API int luaTL_gemm_ex(const void* A, const void* B, void* C,
                            const void* bias, const void* gamma,
                            const float* rowscale,
                            int M, int N, int K, int dtype,
                            long long a_rs, long long a_cs,
                            long long b_rs, long long b_cs,
                            long long c_rs, long long c_cs,
                            long long a_bs, long long b_bs, long long c_bs,
                            int batch, float alpha, float beta, float eps,
                            int pre_op, int post_op)
{
    if (!A || !B || !C) return LUATL_ERR_NULL;
    if (M <= 0 || N <= 0 || K <= 0) return LUATL_ERR_SHAPE;
    const int rc = LTL_DISPATCH(dtype, luaTL_gemm_ex_launch,
        A, B, C, bias, gamma, rowscale, M, N, K,
        a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a_bs, b_bs, c_bs,
        batch, alpha, beta, eps, pre_op, post_op, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("gemm_ex");
}

LUATL_API int luaTL_rmsnorm_bwd(const void* x, const void* gamma, const void* dy,
                                void* dx, float* dgamma,
                                int rows, int cols, float eps, int dtype)
{
    if (!x || !dy || !dx) return LUATL_ERR_NULL;
    if (rows <= 0 || cols <= 0) return LUATL_ERR_SHAPE;
    const int rc = LTL_DISPATCH(dtype, luaTL_rmsnorm_bwd_launch,
        x, gamma, dy, dx, dgamma, rows, cols, eps, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("rmsnorm_bwd");
}

LUATL_API int luaTL_softmax_ex(const void* x, const void* bias, void* out,
                               int rows, int cols, float scale,
                               int causal, int group, int mask_offset, int dtype)
{
    if (!x || !out) return LUATL_ERR_NULL;
    if (rows <= 0 || cols <= 0) return LUATL_ERR_SHAPE;
    const int rc = LTL_DISPATCH(dtype, luaTL_softmax_ex_launch,
        x, bias, out, rows, cols, scale, causal, group, mask_offset, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("softmax_ex");
}

LUATL_API int luaTL_softmax_bwd(const void* y, const void* dy, void* dx,
                                int rows, int cols, float scale,
                                int causal, int group, int mask_offset, int dtype)
{
    if (!y || !dy || !dx) return LUATL_ERR_NULL;
    const int rc = LTL_DISPATCH(dtype, luaTL_softmax_bwd_launch,
        y, dy, dx, rows, cols, scale, causal, group, mask_offset, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("softmax_bwd");
}

LUATL_API int luaTL_swiglu_fwd(const void* a, const void* b, void* out,
                               int rows, int cols, int lda, int ldb, int ldo,
                               int dtype)
{
    if (!a || !b || !out) return LUATL_ERR_NULL;
    const int rc = LTL_DISPATCH(dtype, luaTL_swiglu_fwd_launch,
        a, b, out, rows, cols, lda, ldb, ldo, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("swiglu_fwd");
}

LUATL_API int luaTL_swiglu_bwd(const void* a, const void* b, const void* dout,
                               void* da, void* db,
                               int rows, int cols, int lda, int ldb, int ldo,
                               int dtype)
{
    if (!a || !b || !dout) return LUATL_ERR_NULL;
    const int rc = LTL_DISPATCH(dtype, luaTL_swiglu_bwd_launch,
        a, b, dout, da, db, rows, cols, lda, ldb, ldo, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("swiglu_bwd");
}

LUATL_API int luaTL_rope(void* x, const void* pos_ids,
                         int rows, int heads, int head_dim, int row_stride,
                         int pos_offset, float theta,
                         int interleaved, int inverse, int dtype)
{
    if (!x) return LUATL_ERR_NULL;
    if (rows <= 0 || heads <= 0 || head_dim <= 0) return LUATL_ERR_SHAPE;
    if (row_stride <= 0) row_stride = heads * head_dim;
    const int rc = LTL_DISPATCH(dtype, luaTL_rope_launch,
        x, (const int32_t*)pos_ids, rows, heads, head_dim, row_stride,
        pos_offset, theta, interleaved, inverse, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("rope");
}

LUATL_API int luaTL_embed_fwd(const void* ids, const void* table, void* out,
                              int rows, int dim, int vocab, int dtype)
{
    if (!ids || !table || !out) return LUATL_ERR_NULL;
    const int rc = LTL_DISPATCH(dtype, luaTL_embed_fwd_launch,
        ids, table, out, rows, dim, vocab, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("embed_fwd");
}

LUATL_API int luaTL_embed_bwd(const void* ids, const void* dout, float* dtable,
                              int rows, int dim, int vocab, float scale, int dtype)
{
    if (!ids || !dout || !dtable) return LUATL_ERR_NULL;
    const int rc = LTL_DISPATCH(dtype, luaTL_embed_bwd_launch,
        ids, dout, (void*)dtable, rows, dim, vocab, scale, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("embed_bwd");
}

LUATL_API int luaTL_cross_entropy(const void* logits, const void* targets,
                                  float* losses, void* dlogits,
                                  int rows, int vocab, int ignore_index,
                                  float grad_scale, int dtype)
{
    if (!logits || !targets) return LUATL_ERR_NULL;
    if (rows <= 0 || vocab <= 0) return LUATL_ERR_SHAPE;
    const int rc = LTL_DISPATCH(dtype, luaTL_ce_launch,
        logits, targets, losses, dlogits, rows, vocab, ignore_index,
        grad_scale, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("cross_entropy");
}

LUATL_API int luaTL_reduce_cols(const void* x, float* out, int rows, int cols,
                                long long row_stride, float alpha, float beta,
                                int dtype)
{
    if (!x || !out) return LUATL_ERR_NULL;
    if (row_stride <= 0) row_stride = cols;
    const int rc = LTL_DISPATCH(dtype, luaTL_reduce_cols_launch,
        x, out, rows, cols, row_stride, alpha, beta, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("reduce_cols");
}

LUATL_API int luaTL_l2_accum(const void* x, uint64_t n, float* out, int dtype)
{
    if (!x || !out || n == 0) return LUATL_ERR_NULL;
    const int rc = LTL_DISPATCH(dtype, luaTL_l2_launch, x, n, out, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("l2_accum");
}

LUATL_API int luaTL_clip_by_nsq(void* x, uint64_t n, const float* nsq,
                                float max_norm, int dtype)
{
    if (!x || !nsq || n == 0) return LUATL_ERR_NULL;
    const int rc = LTL_DISPATCH(dtype, luaTL_clip_launch,
        x, n, nsq, max_norm, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("clip");
}

LUATL_API int luaTL_sum_f32(const float* x, uint64_t n, float* out, float scale)
{
    if (!x || !out || n == 0) return LUATL_ERR_NULL;
    luaTL_sum_f32_kernel<<<luaTL_grid1d(n, LUATL_BLOCK_1D),
                           LUATL_BLOCK_1D>>>(x, n, out, scale);
    return luaTL_launch_sync("sum_f32");
}

LUATL_API int luaTL_argmax_rows(const void* x, void* out_i32,
                                int rows, int cols, int dtype)
{
    if (!x || !out_i32) return LUATL_ERR_NULL;
    const int rc = LTL_DISPATCH(dtype, luaTL_argmax_launch,
        x, out_i32, rows, cols, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("argmax");
}

LUATL_API int luaTL_gather_rows(const void* src, const void* idx, void* dst,
                                int n, int cols, int src_rows, int dtype)
{
    if (!src || !idx || !dst) return LUATL_ERR_NULL;
    const int rc = LTL_DISPATCH(dtype, luaTL_gather_launch,
        src, idx, dst, n, cols, src_rows, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("gather_rows");
}

LUATL_API int luaTL_row_rms(const void* x, float* out, int rows, int cols,
                            long long rs, long long cs, float eps, int dtype)
{
    if (!x || !out) return LUATL_ERR_NULL;
    if (rs <= 0) rs = cols;
    if (cs <= 0) cs = 1;
    const int rc = LTL_DISPATCH(dtype, luaTL_rowrms_launch,
        x, out, rows, cols, rs, cs, eps, (cudaStream_t)0);
    if (rc != LUATL_OK) return rc;
    return luaTL_launch_sync("row_rms");
}

/* ---- upload/download of int32 index buffers (tokens, targets) ------- */
LUATL_API int luaTL_upload_i32(void* dev, const int32_t* host, uint64_t n)
{
    if (!dev || !host || n == 0) return LUATL_ERR_NULL;
    LUATL_CHECK("cudaMemcpy(i32 H2D)",
                cudaMemcpy(dev, host, (size_t)n * 4, cudaMemcpyHostToDevice));
    return LUATL_OK;
}

LUATL_API int luaTL_download_i32(const void* dev, int32_t* host, uint64_t n)
{
    if (!dev || !host || n == 0) return LUATL_ERR_NULL;
    LUATL_CHECK("cudaMemcpy(i32 D2H)",
                cudaMemcpy(host, dev, (size_t)n * 4, cudaMemcpyDeviceToHost));
    return LUATL_OK;
}

LUATL_API int luaTL_download_f32(const void* dev, float* host, uint64_t n)
{
    if (!dev || !host || n == 0) return LUATL_ERR_NULL;
    LUATL_CHECK("cudaMemcpy(f32 D2H)",
                cudaMemcpy(host, dev, (size_t)n * 4, cudaMemcpyDeviceToHost));
    return LUATL_OK;
}

/* ---- SAFE teardown --------------------------------------------------
 * Use THIS instead of luaTL_shutdown()/luaTL_shutdown_safe() from Lua.
 * It releases all device memory but deliberately keeps the host-side
 * pool bookkeeping alive, so that a Lua __gc finalizer that fires after
 * teardown finds in_use == 0 and returns harmlessly instead of touching
 * freed host memory.  The residual host allocation is a few KB and is
 * reclaimed by the OS at process exit.
 * ------------------------------------------------------------------ */
LUATL_API void luaTL_finalize(void)
{
    g_epoch++;
    if (g_pool) {
        for (int i = 0; i < g_pool->nblk; ++i) {
            if (g_pool->blks[i].ptr) cudaFree(g_pool->blks[i].ptr);
            g_pool->blks[i].ptr    = NULL;
            g_pool->blks[i].alive  = 0;
            g_pool->blks[i].in_use = 0;   /* makes late frees a no-op */
        }
        for (int b = 0; b < LUATL_NBUCKETS; ++b) g_pool->head[b] = -1;
        g_pool->st.bytes_in_use   = 0;
        g_pool->st.bytes_reserved = 0;
    }
    if (g_rowscale) { cudaFree(g_rowscale); g_rowscale = NULL; g_rowscale_cap = 0; }
    if (g_stage) { free(g_stage); g_stage = NULL; g_stage_size = 0; }
    cudaGetLastError();
    g_initialized = 0;
}

} /* extern "C" */

/* =====================================================================
 *  SECTION T11 :: Program buffer + CUDA Graph capture
 * ================================================================== */

/* ---- void* adapters so LTL_DISPATCH can reach core's typed launchers */
template <typename T>
static int luaTL_transpose_launch_v(const void* in, void* out, int rows, int cols, cudaStream_t s)
{ return luaTL_transpose_launch<T>((const T*)in, (T*)out, rows, cols, s); }

template <typename T>
static int luaTL_bcast_add_launch_v(const void* x, const void* v, void* out, int rows, int cols, float a, cudaStream_t s)
{ return luaTL_bcast_add_launch<T>((const T*)x, (const T*)v, (T*)out, rows, cols, a, s); }

template <typename T>
static int luaTL_ew_launch_v(const void* A, const void* B, void* C, uint64_t n, int op, float alpha, float beta, cudaStream_t s)
{ return luaTL_ew_launch<T>((const T*)A, (const T*)B, (T*)C, n, op, alpha, beta, s); }

template <typename T>
static int luaTL_rmsnorm_launch_v(const void* x, const void* w, void* out, int rows, int cols, float eps, cudaStream_t s)
{ return luaTL_rmsnorm_launch<T>((const T*)x, (const T*)w, (T*)out, rows, cols, eps, s); }

static int luaTL_exec_op(const luaTL_op_t* c, cudaStream_t s)

{
    switch (c->op) {

    case LTL_P_NOP: return LUATL_OK;

    case LTL_P_GEMM:
        if (c->i0 <= 0 || c->i1 <= 0 || c->i2 <= 0) return LUATL_ERR_SHAPE;
        return LTL_DISPATCH(c->dtype, luaTL_gemm_ex_launch,
            c->p0, c->p1, c->p2, c->p3, c->p4, (const float*)c->p5,
            c->i0, c->i1, c->i2,
            c->s0, c->s1, c->s2, c->s3, c->s4, c->s5, c->s6, c->s7, c->s8,
            (c->i3 > 0 ? c->i3 : 1), c->f0, c->f1, c->f2, c->pre, c->act, s);

    case LTL_P_EW:
        if (c->n == 0) return LUATL_OK;
        return LTL_DISPATCH(c->dtype, luaTL_ew_launch_v,
            (const void*)c->p0, (const void*)c->p1, c->p2, c->n,
            c->i0, c->f0, c->f1, s);

    case LTL_P_RMSNORM:
        return LTL_DISPATCH(c->dtype, luaTL_rmsnorm_launch_v,
            (const void*)c->p0, (const void*)c->p1, c->p2,
            c->i0, c->i1, c->f0, s);

    case LTL_P_RMSNORM_BWD:
        return LTL_DISPATCH(c->dtype, luaTL_rmsnorm_bwd_launch,
            c->p0, c->p1, c->p2, c->p3, (float*)c->p4,
            c->i0, c->i1, c->f0, s);

    case LTL_P_SOFTMAX:
        return LTL_DISPATCH(c->dtype, luaTL_softmax_ex_launch,
            c->p0, c->p1, c->p2, c->i0, c->i1, c->f0, c->i2, c->i3, c->i4, s);

    case LTL_P_SOFTMAX_BWD:
        return LTL_DISPATCH(c->dtype, luaTL_softmax_bwd_launch,
            c->p0, c->p1, c->p2, c->i0, c->i1, c->f0, c->i2, c->i3, c->i4, s);

    case LTL_P_SWIGLU:
        return LTL_DISPATCH(c->dtype, luaTL_swiglu_fwd_launch,
            c->p0, c->p1, c->p2, c->i0, c->i1, c->i2, c->i3, c->i4, s);

    case LTL_P_SWIGLU_BWD:
        return LTL_DISPATCH(c->dtype, luaTL_swiglu_bwd_launch,
            c->p0, c->p1, c->p2, c->p3, c->p4,
            c->i0, c->i1, c->i2, c->i3, c->i4, s);

    case LTL_P_ROPE:
        return LTL_DISPATCH(c->dtype, luaTL_rope_launch,
            c->p0, (const int32_t*)c->p1, c->i0, c->i1, c->i2,
            (c->i3 > 0 ? c->i3 : c->i1 * c->i2), c->i4, c->f0, c->i5, c->i6, s);

    case LTL_P_EMBED:
        return LTL_DISPATCH(c->dtype, luaTL_embed_fwd_launch,
            c->p0, c->p1, c->p2, c->i0, c->i1, c->i2, s);

    case LTL_P_EMBED_BWD:
        return LTL_DISPATCH(c->dtype, luaTL_embed_bwd_launch,
            c->p0, c->p1, c->p2, c->i0, c->i1, c->i2, c->f0, s);

    case LTL_P_CE:
        return LTL_DISPATCH(c->dtype, luaTL_ce_launch,
            c->p0, c->p1, (float*)c->p2, c->p3,
            c->i0, c->i1, c->i2, c->f0, s);

    case LTL_P_REDUCE_COLS:
        return LTL_DISPATCH(c->dtype, luaTL_reduce_cols_launch,
            c->p0, (float*)c->p2, c->i0, c->i1,
            (c->s0 > 0 ? c->s0 : c->i1), c->f0, c->f1, s);

    case LTL_P_L2:
        if (c->n == 0) return LUATL_OK;
        return LTL_DISPATCH(c->dtype, luaTL_l2_launch,
            c->p0, c->n, (float*)c->p2, s);

    case LTL_P_CLIP:
        if (c->n == 0) return LUATL_OK;
        return LTL_DISPATCH(c->dtype, luaTL_clip_launch,
            c->p2, c->n, (const float*)c->p1, c->f0, s);

    case LTL_P_ADAMW: {
        if (c->n == 0) return LUATL_OK;
        const int step = (c->i0 < 1) ? 1 : c->i0;
        const float bc1 = 1.0f / (1.0f - powf(c->f1, (float)step));
        const float bc2 = 1.0f / (1.0f - powf(c->f2, (float)step));
        const unsigned g = luaTL_grid1d(c->n, LUATL_BLOCK_1D);
        if (c->dtype == LUATL_F16)
            luaTL_adamw_kernel<__half><<<g, LUATL_BLOCK_1D, 0, s>>>(
                (float*)c->p0, (const __half*)c->p1, (float*)c->p2, (float*)c->p3,
                (__half*)c->p4, c->n, c->f0, c->f1, c->f2, c->f3, c->f4,
                bc1, bc2, c->f5);
        else if (c->dtype == LUATL_BF16)
            luaTL_adamw_kernel<luatl_bf16><<<g, LUATL_BLOCK_1D, 0, s>>>(
                (float*)c->p0, (const luatl_bf16*)c->p1, (float*)c->p2, (float*)c->p3,
                (luatl_bf16*)c->p4, c->n, c->f0, c->f1, c->f2, c->f3, c->f4,
                bc1, bc2, c->f5);
        else
            luaTL_adamw_kernel<float><<<g, LUATL_BLOCK_1D, 0, s>>>(
                (float*)c->p0, (const float*)c->p1, (float*)c->p2, (float*)c->p3,
                (float*)c->p4, c->n, c->f0, c->f1, c->f2, c->f3, c->f4,
                bc1, bc2, c->f5);
        return luaTL_launch_async("adamw");
    }

    case LTL_P_ZERO:
        if (!c->p2 || c->n == 0) return LUATL_OK;
        LUATL_CHECK("cudaMemsetAsync", cudaMemsetAsync(c->p2, 0, (size_t)c->n, s));
        return LUATL_OK;

    case LTL_P_COPY:
        if (!c->p0 || !c->p2 || c->n == 0) return LUATL_OK;
        LUATL_CHECK("cudaMemcpyAsync",
                    cudaMemcpyAsync(c->p2, c->p0, (size_t)c->n,
                                    cudaMemcpyDeviceToDevice, s));
        return LUATL_OK;

    case LTL_P_CAST:
        return luaTL_cast_launch(c->p0, c->i0, c->p2, c->i1, c->n, s);

    case LTL_P_TRANSPOSE:
        return LTL_DISPATCH(c->dtype, luaTL_transpose_launch_v,
                            c->p0, c->p2, c->i0, c->i1, s);

    case LTL_P_BCAST_ADD:
        return LTL_DISPATCH(c->dtype, luaTL_bcast_add_launch_v,
                            c->p0, c->p1, c->p2, c->i0, c->i1, c->f0, s);

    case LTL_P_ARGMAX:
        return LTL_DISPATCH(c->dtype, luaTL_argmax_launch,
                            c->p0, c->p2, c->i0, c->i1, s);

    case LTL_P_SUM:
        if (c->n == 0) return LUATL_OK;
        luaTL_sum_f32_kernel<<<luaTL_grid1d(c->n, LUATL_BLOCK_1D),
                               LUATL_BLOCK_1D, 0, s>>>(
            (const float*)c->p0, c->n, (float*)c->p2, c->f0);
        return luaTL_launch_async("sum");

    case LTL_P_GATHER:
        return LTL_DISPATCH(c->dtype, luaTL_gather_launch,
                            c->p0, c->p1, c->p2, c->i0, c->i1, c->i2, s);

    case LTL_P_ROWRMS:
        return LTL_DISPATCH(c->dtype, luaTL_rowrms_launch,
            c->p0, (float*)c->p2, c->i0, c->i1,
            (c->s0 > 0 ? c->s0 : c->i1), (c->s1 > 0 ? c->s1 : 1), c->f0, s);

    case LTL_P_SYNC:
        LUATL_CHECK("cudaStreamSynchronize", cudaStreamSynchronize(s));
        return LUATL_OK;

    default:
        luaTL_seterr("prog: unknown opcode %d", c->op);
        return LUATL_ERR_ARG;
    }
}

extern "C" {

LUATL_API luaTL_prog_t* luaTL_prog_create(int capacity, int timing)
{
    if (capacity <= 0) capacity = 4096;
    luaTL_prog_t* P = (luaTL_prog_t*)calloc(1, sizeof(luaTL_prog_t));
    if (!P) { luaTL_seterr("prog_create: host OOM"); return NULL; }
    P->ops = (luaTL_op_t*)calloc((size_t)capacity, sizeof(luaTL_op_t));
    if (!P->ops) { free(P); luaTL_seterr("prog_create: host OOM"); return NULL; }
    P->capacity = capacity;

    cudaStream_t s = NULL;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        free(P->ops); free(P);
        luaTL_seterr("prog_create: cudaStreamCreate failed");
        return NULL;
    }
    P->stream      = (void*)s;
    P->owns_stream = 1;
    P->timing      = timing ? 1 : 0;
    P->flags       = 1;              /* bit0: synchronize at end */
    P->last_ms     = -1.0;
    if (P->timing) {
        cudaEvent_t a = NULL, b = NULL;
        cudaEventCreate(&a); cudaEventCreate(&b);
        P->ev_start = (void*)a; P->ev_stop = (void*)b;
    }
    return P;
}

LUATL_API void luaTL_prog_graph_free(luaTL_prog_t* P)
{
    if (!P) return;
    if (P->graph_exec) { cudaGraphExecDestroy((cudaGraphExec_t)P->graph_exec);
                         P->graph_exec = NULL; }
    if (P->graph)      { cudaGraphDestroy((cudaGraph_t)P->graph);
                         P->graph = NULL; }
    P->captured = 0;
    cudaGetLastError();
}

LUATL_API void luaTL_prog_destroy(luaTL_prog_t* P)
{
    if (!P) return;
    luaTL_prog_graph_free(P);
    if (P->ev_start) cudaEventDestroy((cudaEvent_t)P->ev_start);
    if (P->ev_stop)  cudaEventDestroy((cudaEvent_t)P->ev_stop);
    if (P->stream && P->owns_stream) cudaStreamDestroy((cudaStream_t)P->stream);
    free(P->ops);
    free(P);
    cudaGetLastError();
}

LUATL_API void luaTL_prog_reset(luaTL_prog_t* P)
{
    if (P) { P->count = 0; P->status = LUATL_OK; }
}

/* Execute the op list directly on the stream (no graph). */
LUATL_API int luaTL_prog_run(luaTL_prog_t* P)
{
    if (!P) return LUATL_ERR_NULL;
    if (P->count <= 0) { P->status = LUATL_OK; return LUATL_OK; }
    if (P->count > P->capacity) {
        luaTL_seterr("prog_run: count %d > capacity %d", P->count, P->capacity);
        P->status = LUATL_ERR_CAPACITY;
        return LUATL_ERR_CAPACITY;
    }
    cudaStream_t s = (cudaStream_t)P->stream;

    if (P->timing && P->ev_start) cudaEventRecord((cudaEvent_t)P->ev_start, s);

    for (int i = 0; i < P->count; ++i) {
        const int rc = luaTL_exec_op(&P->ops[i], s);
        if (rc != LUATL_OK) {
            luaTL_seterr("prog_run: op %d (opcode %d) failed: %s",
                         i, P->ops[i].op, g_err);
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
            P->status = luaTL_cuda_fail("prog_run: stream sync", e);
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

/* Capture the current op list into a CUDA graph.
 * PRE-REQUISITE: run the program normally at least once first, so that
 * any lazy scratch (RMS row-scale buffer, autotune cache) is warm --
 * cudaMalloc is illegal during capture. */
LUATL_API int luaTL_prog_capture(luaTL_prog_t* P)
{
    if (!P) return LUATL_ERR_NULL;
    if (P->count <= 0) return LUATL_ERR_ARG;

    luaTL_prog_graph_free(P);
    cudaStream_t s = (cudaStream_t)P->stream;

    LUATL_CHECK("cudaStreamSynchronize(pre-capture)", cudaStreamSynchronize(s));

    cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) return luaTL_cuda_fail("cudaStreamBeginCapture", e);
    P->capturing = 1;

    int rc = LUATL_OK;
    for (int i = 0; i < P->count; ++i) {
        if (P->ops[i].op == LTL_P_SYNC) continue;   /* illegal while capturing */
        rc = luaTL_exec_op(&P->ops[i], s);
        if (rc != LUATL_OK) break;
    }

    cudaGraph_t g = NULL;
    cudaError_t e2 = cudaStreamEndCapture(s, &g);
    P->capturing = 0;
    if (rc != LUATL_OK) {
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();
        return rc;
    }
    if (e2 != cudaSuccess) return luaTL_cuda_fail("cudaStreamEndCapture", e2);

    cudaGraphExec_t ge = NULL;
#if CUDART_VERSION >= 12000
    e = cudaGraphInstantiate(&ge, g, 0);
#else
    e = cudaGraphInstantiate(&ge, g, NULL, NULL, 0);
#endif
    if (e != cudaSuccess) { cudaGraphDestroy(g);
                            return luaTL_cuda_fail("cudaGraphInstantiate", e); }

    P->graph      = (void*)g;
    P->graph_exec = (void*)ge;
    P->captured   = 1;
    return LUATL_OK;
}

/* Replay the captured graph: ONE driver call for the whole step. */
LUATL_API int luaTL_prog_replay(luaTL_prog_t* P)
{
    if (!P) return LUATL_ERR_NULL;
    if (!P->captured || !P->graph_exec) return luaTL_prog_run(P);

    cudaStream_t s = (cudaStream_t)P->stream;
    if (P->timing && P->ev_start) cudaEventRecord((cudaEvent_t)P->ev_start, s);

    cudaError_t e = cudaGraphLaunch((cudaGraphExec_t)P->graph_exec, s);
    if (e != cudaSuccess) { P->status = luaTL_cuda_fail("cudaGraphLaunch", e);
                            return P->status; }

    if (P->timing && P->ev_stop) cudaEventRecord((cudaEvent_t)P->ev_stop, s);

    if (P->flags & 1) {
        e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) { P->status = luaTL_cuda_fail("graph sync", e);
                                return P->status; }
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

LUATL_API int luaTL_prog_wait(luaTL_prog_t* P)
{
    if (!P || !P->stream) return LUATL_ERR_NULL;
    LUATL_CHECK("cudaStreamSynchronize", cudaStreamSynchronize((cudaStream_t)P->stream));
    if (P->timing && P->ev_stop) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, (cudaEvent_t)P->ev_start,
                                 (cudaEvent_t)P->ev_stop) == cudaSuccess)
            P->last_ms = (double)ms;
    }
    return LUATL_OK;
}

LUATL_API int luaTL_prog_push(luaTL_prog_t* P, const luaTL_op_t* op)
{
    if (!P || !op) return LUATL_ERR_NULL;
    if (P->count >= P->capacity) return LUATL_ERR_CAPACITY;
    P->ops[P->count++] = *op;
    return LUATL_OK;
}

} /* extern "C" */