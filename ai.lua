package.path = package.path
    .. ";./ai/core/?.lua"
    .. ";./ai/nn/?.lua"
    .. ";./ai/data/?.lua"
    .. ";./ai/optim/?.lua"
    .. ";./ai/gpu/?.lua"
    .. ";./tests/?.lua"

local ai = {}

local function safe_require(name)
    local ok, lib = pcall(require, name)
    if ok then return lib end
    return nil
end

local modules = {
    {"Matrix", "matrix"},
    {"Tensor", "tensor"},
    {"Tensor2", "tensor2"},
    {"Tokenizer", "tokenizer"},
    {"Data", "data"},
    {"Embedding", "embedding"},
    {"Positional", "positional"},
    {"Transformer", "transformer"},
    {"NN", "nn"},
    {"Optim", "optim"},
    {"Optim2", "optim2"},
    {"Train", "train"},
    {"LMTrain", "lmtrain"},
    {"GPU", "gpu"},
    {"BLAS", "blas"},
    {"LuaTL", "luatl_adapter"},
    {"RoPE", "rope"}
}

for _, m in ipairs(modules) do
    local upper_name, file_name = m[1], m[2]
    local ok, lib = pcall(require, file_name)
    if ok then
        ai[upper_name] = lib
        ai[file_name]  = lib
    else
        print("[LanternL] Warning: Failed to load module '" .. file_name .. "': " .. tostring(lib))
    end
end

function ai.profile()
    local luaTL = ai.LuaTL
    if not luaTL or not luaTL.available then
        print("Profiling: luaTL (GPU) is not active. Running in pure CPU mode.")
        return
    end

    local p = luaTL.get_profile()
    print("=========================================")
    print("  luaTL GPU Profiler")
    print("=========================================")
    print(string.format("  VRAM Allocations : %d", p.allocs))
    print(string.format("  Host -> Device   : %.4f sec", p.h2d))
    print(string.format("  Device -> Host   : %.4f sec", p.d2h))
    print(string.format("  Compute Kernels  : %.4f sec", p.compute))
    print("=========================================")
    if p.h2d > p.compute * 2 then
        print("  ⚠️ Warning: Data transfer is bottlenecking compute.")
        print("     Consider using larger batch sizes.")
    end
end

function ai.benchmark()
    local matrix = ai.Matrix
    print("=========================================")
    print("  LanternL CPU/GPU Benchmark")
    print("=========================================")
    
    -- 1024x1024 matrices (roughly 4MB each)
    local size = 512 
    print("Generating " .. size .. "x" .. size .. " matrices...")
    local A = matrix.random(size, size)
    local B = matrix.random(size, size)
    
    -- 1. CPU Benchmark
    matrix.use("lua")
    local t0 = os.clock()
    local C_cpu = matrix.multiply(A, B)
    local cpu_time = os.clock() - t0
    print(string.format("  CPU (Pure LuaJIT) : %.4f sec", cpu_time))
    
    -- 2. GPU Benchmark (if available)
    local gpu_time = "N/A"
    if ai.LuaTL and ai.LuaTL.available then
        matrix.use("luatl_adapter")
        local t1 = os.clock()
        local C_gpu = matrix.multiply(A, B)
        gpu_time = os.clock() - t1
        print(string.format("  GPU (luaTL)       : %.4f sec", gpu_time))
        
        -- Verify results match
        if matrix.equals(C_cpu, C_gpu, 0.1) then
            print("  Verification       : PASSED (CPU ~= GPU)")
        else
            print("  Verification       : MISMATCH (Check math)")
        end
        matrix.use("auto") -- reset to auto
    else
        print("  GPU (luaTL)       : Not installed/available")
    end
    
    print("=========================================")
end

function ai.help()
    print([=[
1. QUICKSTART (Text to AI in 5 lines)
--------------------------------------------------------------------
local ai = require("ai")

local model = ai.LMTrain {
    data = "hello world, this is lua ai!",
    preset = "auto",   -- Auto-sizes the model dimensions
    epochs = 300
}

model:run()             -- Watch the progress bar!
print(model:generate("hello", 10)) -- Generate text!


2. TRAINING WITH A TOKENIZER (For larger texts)
--------------------------------------------------------------------
-- Train a BPE Tokenizer on your text
local tok = ai.Tokenizer.new { text = "your long text here", vocab_size = 500 }
tok:train()

-- Pass the tokenizer to LMTrain! It handles encoding/decoding automatically.
local trainer = ai.LMTrain {
    data = "your long text here",
    tokenizer = tok,
    epochs = 100
}
trainer:run()
print(trainer:generate("your", 20))


3. DOWNLOADING DATASETS (HuggingFace)
--------------------------------------------------------------------
-- Auto-discovers files, parses Parquet/CSV/TXT, limits to 500MB
local dataset = ai.Data("your_username/your_dataset", "500MB")

local tok = ai.Tokenizer.new { text_file = dataset.files[1], vocab_size = 1000 }
tok:train()

dataset:config { batch_size = 8, tokenizer = tok }
local batches = dataset:batches()


4. ADVANCED TRAINING & EARLY STOPPING
--------------------------------------------------------------------
local trainer = ai.LMTrain {
    data = "your data here",
    epochs = 1000,
    lr = 0.01,           -- SGD learning rate
    lr_decay = true,     -- Slow down learning over time
    stop_loss = 0.05,    -- Stop if loss hits 0.05
    patience = 50,       -- Stop if no improvement for 50 epochs
    max_seq = 256,       -- Context window size
    bar_style = 2        -- 1=#, 2=blocks, 3==
}

-- Custom Logging (access internal state)
trainer:config {
    log = function(t)
        print(string.format("Epoch %d | Acc: %.1f%%", t.epoch, t.accuracy))
    end
}

trainer:run()


5. SAVE, LOAD, AND PUSH TO HUGGINGFACE
--------------------------------------------------------------------
trainer:save("my_lua_model.txt")
trainer:load("my_lua_model.txt")

-- Push to HuggingFace Hub (Requires running 'hf auth login' in your terminal first)
trainer:push("your_username/your_model_repo")


6. CHECK GPU / HARDWARE STATUS
--------------------------------------------------------------------
for _, b in ipairs(ai.Matrix.backends()) do
    print(string.format("Backend: %-15s | Available: %-5s | Reason: %s", 
          b.name, tostring(b.available), b.reason))
end


7. CORE MODULES (ai.X)
--------------------------------------------------------------------
  ai.LMTrain      : High-level Trainer & Generator
  ai.Data         : Dataset downloader & batcher
  ai.Tokenizer    : BPE Tokenizer (train, encode, decode)
  ai.Transformer  : Raw Transformer architecture (RoPE, Causal, RMSNorm)
  ai.Matrix       : CPU/GPU Matrix math engine
  ai.Optim2       : Optimizers (Currently SGD)

Type `ai.h()` to see this menu anytime!
====================================================================
]=])
end

ai.h = ai.help

return ai
