package.path = package.path
    .. ";./ai/core/?.lua"
    .. ";./ai/nn/?.lua"
    .. ";./ai/data/?.lua"
    .. ";./ai/optim/?.lua"

local ai = {}

local function safe_require(name)
    local ok, lib = pcall(require, name)
    if ok then return lib end
    return nil
end

local modules = {
    Tokenizer   = "tokenizer",
    Data        = "data",
    Matrix      = "matrix",
    Tensor      = "tensor",
    Tensor2     = "tensor2",
    Embedding   = "embedding",
    Positional  = "positional",
    Linear      = "linear",
    Attention   = "attention",
    RMSNorm     = "rmsnorm",
    SwiGLU      = "swiglu",
    Block       = "block",
    Transformer = "transformer",
    Model       = "model",
    NN          = "nn",
    Optim       = "optim",
    Optim2      = "optim2",
    Train       = "train",
    LMTrain     = "lmtrain",
    GPU         = "gpu",
    BLAS        = "blas",
    LuaTL       = "luatl_adapter",
}

for upper_name, file_name in pairs(modules) do
    local lib = safe_require(file_name)
    ai[upper_name] = lib
    ai[file_name]  = lib
end

function ai.help()
    -- Using [=[ ]=] so we can use [[ ]] inside the string if needed
    print([=[
====================================================================
  🏮 LANTERNL: The Pure-Lua AI Ecosystem
  No Python. No bloat. Just LuaJIT and CUDA.
====================================================================

1. THE 5-SECOND QUICKSTART (Text to AI)
--------------------------------------------------------------------
local ai = require("ai")

-- Pass a string, auto-size the AI, train, and generate!
local model = ai.LMTrain {
    data = "hello world, this is lua ai!",
    preset = "auto",   -- Automatically picks model dimensions
    epochs = 300
}

model:run() -- Watch the progress bar!

-- Give it a seed, and it generates text!
print( model:generate("hello", 10) )


2. DOWNLOADING REAL DATA (HuggingFace)
--------------------------------------------------------------------
-- Auto-discovers files, reads Parquet/CSV/TXT, limits to 500MB
local dataset = ai.Data("your_username/your_dataset", "500MB")

-- Config your batch size and attach a tokenizer
local tok = ai.Tokenizer.new { text_file = dataset.files[1], vocab_size = 1000 }
tok:train()

dataset:config { batch_size = 8, tokenizer = tok }
local batches = dataset:batches()


3. ADVANCED TRAINING & EARLY STOPPING
--------------------------------------------------------------------
local trainer = ai.LMTrain {
    data = "your data here",
    epochs = 1000,
    lr = 0.01,
    lr_decay = true,     -- Slow down learning over time
    stop_loss = 0.05,    -- Stop if loss hits 0.05
    patience = 50,       -- Stop if no improvement for 50 epochs
    bar_style = 2,       -- 1=#, 2=▓▒, 3==
    every = 10           -- Print progress every 10 epochs
}

-- Custom Logging (access internal state)
trainer:config {
    log = function(t)
        print(string.format("Epoch %d | Acc: %.1f%%", t.epoch, t.accuracy))
    end
}

trainer:run()


4. SAVE, LOAD, AND PUSH TO HUGGINGFACE
--------------------------------------------------------------------
trainer:save("my_lua_model.txt")
trainer:load("my_lua_model.txt")

-- Push your trained model back to HuggingFace Hub!
trainer:push("your_username/your_model_repo", "hf_your_token_here")


5. CHECK GPU / HARDWARE STATUS
--------------------------------------------------------------------
-- See if your CUDA (luaTL) or BLAS backends loaded successfully
for _, b in ipairs(ai.Matrix.backends()) do
    print(string.format("Backend: %-15s | Available: %-5s | Reason: %s", 
          b.name, tostring(b.available), b.reason))
end


6. CORE MODULES REFERENCE (ai.X)
--------------------------------------------------------------------
  ai.LMTrain      : High-level Trainer & Generator
  ai.Data         : Dataset downloader & batcher
  ai.Tokenizer    : BPE Tokenizer (train, encode, decode)
  ai.Transformer  : Raw Transformer model architecture
  ai.Matrix       : CPU/GPU Matrix math engine
  ai.Optim2       : Optimizers (SGD, AdamW, etc.)

Type `ai.h()` to see this menu anytime!
====================================================================
]=])
end

ai.h = ai.help

return ai