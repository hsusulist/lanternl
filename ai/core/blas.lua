local ok_ffi, ffi = pcall(require, "ffi")
if not ok_ffi then
    return { available = false }
end

local ok_lib, openblas = pcall(ffi.load, "openblas")
if not ok_lib then
    return { available = false }
end

ffi.cdef[[
void cblas_sgemm(int Order, int TransA, int TransB,
                  int M, int N, int K,
                  float alpha, const float *A, int lda,
                  const float *B, int ldb,
                  float beta, float *C, int ldc);
]]

local CblasRowMajor = 101
local CblasNoTrans  = 111

local blas = { available = true }

function blas.matmul(a_flat, m, k, b_flat, n)
    local A = ffi.new("float[?]", m * k)
    local B = ffi.new("float[?]", k * n)
    local C = ffi.new("float[?]", m * n)

    for i = 0, m * k - 1 do A[i] = a_flat[i + 1] end
    for i = 0, k * n - 1 do B[i] = b_flat[i + 1] end

    openblas.cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,
        1.0, A, k,
        B, n,
        0.0, C, n
    )

    local result = {}
    for i = 0, m * n - 1 do result[i + 1] = C[i] end
    return result
end

return blas
