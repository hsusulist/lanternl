--  bench_gpu_transformer.lua — prove the resident path is real
--
--    luajit bench_gpu_transformer.lua              -- default config
--    luajit bench_gpu_transformer.lua 256 4 8 1024 8 64 5000
--                     dim layers heads ffn batch seq vocab
--
--  Reports achieved GFLOP/s for a FULL forward + backward + AdamW step,
--  the round-trip GEMM baseline, and the CPU-autograd wall time.

package.path = package.path .. ";./ai/core/?.lua;./ai/nn/?.lua;./ai/gpu/?.lua;./?.lua"

local lmlog = require("lmlog")
local now   = lmlog.now
local sf    = string.format

local A = { ... }
local dim    = tonumber(A[1]) or 256
local layers = tonumber(A[2]) or 4
local heads  = tonumber(A[3]) or 8
local ffn    = tonumber(A[4]) or 1024
local bs     = tonumber(A[5]) or 8
local seq    = tonumber(A[6]) or 128
local vocab  = tonumber(A[7]) or 4096
local ITERS  = tonumber(A[8]) or 30
local WARM   = 5

local adapter = require("luatl_adapter")
assert(adapter.available, "luaTL not available: " .. tostring(adapter.error))
local hw = adapter.hardware()

print("=====================================================================")
print(sf(" GPU : %s (%s, %d SMs, %.2f TFLOP/s fp32 peak, %.0f GB/s)",
      hw.name, hw.sm, hw.multiprocessors, hw.gflops_fp32 / 1000, hw.bandwidth_gbs))
print(sf(" cfg : dim=%d layers=%d heads=%d ffn=%d batch=%d seq=%d vocab=%d",
      dim, layers, heads, ffn, bs, seq, vocab))
print("=====================================================================")

-- ---------------------------------------------------------------------
--  0. Reference ceiling: a resident GEMM of comparable arithmetic size
-- ---------------------------------------------------------------------
local luaTL = adapter.luaTL
do
    local M, N, K = bs * seq, ffn, dim
    local r = luaTL.benchmark_gemm(M, N, K, "f32", 20)
    print(sf("[0] resident GEMM %dx%dx%d           : %8.1f GFLOP/s",
          M, N, K, r.tflops * 1000))
end

-- ---------------------------------------------------------------------
--  1. The round-tripping path we are replacing (adapter.matmul on tables)
-- ---------------------------------------------------------------------
do
    local M, N, K = bs * seq, ffn, dim
    local a, b = {}, {}
    for i = 1, M * K do a[i] = 0.01 end
    for i = 1, K * N do b[i] = 0.01 end
    adapter.matmul(a, M, K, b, N)                   -- warm
    local t0 = now()
    for _ = 1, 10 do adapter.matmul(a, M, K, b, N) end
    local dt = (now() - t0) / 10
    print(sf("[1] round-trip GEMM (flat tables)    : %8.1f GFLOP/s  (%.2f ms/call)",
          (2 * M * N * K) / dt / 1e9, dt * 1e3))
end

-- ---------------------------------------------------------------------
--  2. Full resident fwd + bwd + AdamW step
-- ---------------------------------------------------------------------
local GT = require("gpu_transformer")
local m = GT.new{
    vocab = vocab, dim = dim, layers = layers, heads = heads, ffn = ffn,
    max_seq = seq, batch_size = bs, causal = true, pos = "rope",
    tie = true, clip_norm = 1.0, weight_decay = 0.01, timing = true,
    want_acc = false,
}
local flops_step, flops_fwd = m:flops_per_step()
print(sf(" model: %s params | %.3f GFLOP/step (fwd %.3f) | VRAM in use %.1f MB",
      lmlog.fmt_count(m:nparams()), flops_step / 1e9, flops_fwd / 1e9,
      m:vram().pool_in_use / 1048576))

-- synthetic windows
local wins = {}
for b = 1, bs do
    local inp, tgt = {}, {}
    for t = 1, seq do
        inp[t] = math.random(1, vocab)
        tgt[t] = math.random(1, vocab)
    end
    wins[b] = { inp, tgt, seq }
end

for i = 1, WARM do m:train_step(wins, bs, 1e-3, i) end
adapter.sync()

local t0 = now()
for i = 1, ITERS do m:train_step(wins, bs, 1e-3, WARM + i) end
adapter.sync()
local wall_ms = (now() - t0) / ITERS * 1e3
local gpu_ms  = m:last_step_ms()

print("---------------------------------------------------------------------")
print(sf("[2] resident fwd+bwd+AdamW  wall        : %8.2f ms/step -> %8.1f GFLOP/s",
      wall_ms, flops_step / (wall_ms * 1e-3) / 1e9))
print(sf("    kernel time (CUDA events, fwd+bwd)  : %8.2f ms/step -> %8.1f GFLOP/s",
      gpu_ms, flops_step / (gpu_ms * 1e-3) / 1e9))
print(sf("    host overhead per step              : %8.2f ms (%.1f%%)",
      wall_ms - gpu_ms, 100 * (wall_ms - gpu_ms) / wall_ms))
print(sf("    throughput                          : %8.0f tokens/s",
      (bs * seq) / (wall_ms * 1e-3)))

-- ---------------------------------------------------------------------
--  3. CUDA graph replay
-- ---------------------------------------------------------------------
local ok, err = m:capture()
if ok then
    for i = 1, WARM do m:train_step(wins, bs, 1e-3, i) end
    adapter.sync()
    local t1 = now()
    for i = 1, ITERS do m:train_step(wins, bs, 1e-3, i) end
    adapter.sync()
    local g_ms = (now() - t1) / ITERS * 1e3
    print(sf("[3] with CUDA graph replay             : %8.2f ms/step -> %8.1f GFLOP/s  (%.2fx)",
          g_ms, flops_step / (g_ms * 1e-3) / 1e9, wall_ms / g_ms))
else
    print("[3] CUDA graph capture unavailable: " .. tostring(err))
end

-- ---------------------------------------------------------------------
--  4. CPU autograd baseline (small config only -- it is slow on purpose)
-- ---------------------------------------------------------------------
print("---------------------------------------------------------------------")
local small = (dim * layers * ffn * bs * seq <= 256 * 2 * 512 * 4 * 64)
if not small then
    print("[4] CPU baseline skipped (config too large; rerun with e.g.")
    print("    luajit bench_gpu_transformer.lua 128 2 4 256 4 64 512 5)")
else
    local okc, LMTrain = pcall(require, "lmtrain")
    if not okc then
        print("[4] CPU baseline unavailable: " .. tostring(LMTrain))
    else
        local Tensor = require("tensor2")
        local Transformer = require("transformer")
        local cm = Transformer.new{
            vocab = vocab, dim = dim, layers = layers, heads = heads,
            ffn = ffn, causal = true, pos = "rope", max_seq = seq, tie = true,
        }
        local w = wins[1]
        Tensor.training = true
        local t2 = now()
        for _ = 1, 3 do
            local lg = cm:forward(w[1])
            local ls = Tensor.cross_entropy(lg, w[2])
            ls:backward()
        end
        local cpu_ms = (now() - t2) / 3 * 1e3
        -- CPU does ONE window per pass; scale to a full batch for fairness
        local cpu_batch_ms = cpu_ms * bs
        print(sf("[4] CPU autograd (tensor2)             : %8.2f ms/window, %.2f ms/batch-of-%d",
              cpu_ms, cpu_batch_ms, bs))
        print(sf("    SPEEDUP (resident GPU vs CPU)      : %8.1fx   (graph: see [3])",
              cpu_batch_ms / wall_ms))
    end
end

print("=====================================================================")
m:free()
adapter.shutdown()
