local ok_ffi, ffi = pcall(require, "ffi")
if not ok_ffi then
    return { available = false }
end

local ok_cudart, cudart = pcall(ffi.load, "cudart")
local ok_cublas, cublas_lib = pcall(ffi.load, "cublas")

if not ok_cudart or not ok_cublas then
    return { available = false }
end

ffi.cdef[[
typedef struct cublasContext *cublasHandle_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 } cublasOperation_t;
typedef enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 } cudaMemcpyKind;

int cudaMalloc(void **devPtr, size_t size);
int cudaFree(void *devPtr);
int cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);

int cublasCreate_v2(cublasHandle_t *handle);
int cublasDestroy_v2(cublasHandle_t handle);
int cublasSgemm_v2(cublasHandle_t handle,
                    cublasOperation_t transa, cublasOperation_t transb,
                    int m, int n, int k,
                    const float *alpha,
                    const float *A, int lda,
                    const float *B, int ldb,
                    const float *beta,
                    float *C, int ldc);
]]

local handle_ptr = ffi.new("cublasHandle_t[1]")
local status = cublas_lib.cublasCreate_v2(handle_ptr)
if status ~= 0 then
    return { available = false }
end
local handle = handle_ptr[0]

local gpu = { available = true }

function gpu.matmul(a_flat, m, k, b_flat, n)
    local size_a = m * k * ffi.sizeof("float")
    local size_b = k * n * ffi.sizeof("float")
    local size_c = m * n * ffi.sizeof("float")

    local h_a = ffi.new("float[?]", m * k)
    local h_b = ffi.new("float[?]", k * n)
    for i = 0, m * k - 1 do h_a[i] = a_flat[i + 1] end
    for i = 0, k * n - 1 do h_b[i] = b_flat[i + 1] end

    local d_a = ffi.new("void*[1]")
    local d_b = ffi.new("void*[1]")
    local d_c = ffi.new("void*[1]")

    cudart.cudaMalloc(d_a, size_a)
    cudart.cudaMalloc(d_b, size_b)
    cudart.cudaMalloc(d_c, size_c)

    cudart.cudaMemcpy(d_a[0], h_a, size_a, 1)
    cudart.cudaMemcpy(d_b[0], h_b, size_b, 1)

    local alpha = ffi.new("float[1]", 1.0)
    local beta = ffi.new("float[1]", 0.0)

    cublas_lib.cublasSgemm_v2(
        handle, 0, 0,
        n, m, k,
        alpha,
        ffi.cast("float*", d_b[0]), n,
        ffi.cast("float*", d_a[0]), k,
        beta,
        ffi.cast("float*", d_c[0]), n
    )

    local h_c = ffi.new("float[?]", m * n)
    cudart.cudaMemcpy(h_c, d_c[0], size_c, 2)

    cudart.cudaFree(d_a[0])
    cudart.cudaFree(d_b[0])
    cudart.cudaFree(d_c[0])

    local result = {}
    for i = 0, m * n - 1 do result[i + 1] = h_c[i] end
    return result
end

return gpu
