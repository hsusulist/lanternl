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
    print([=[
=== lanternl quick guide ===

TOKENIZER:
local tok = ai.Tokenizer.new{ text = "hello world", vocab_size = 60 }
tok:train()
tok:encode("hello")

TRAIN A LANGUAGE MODEL:
local train = ai.LMTrain{ data = "hello world", e = 200 }
train:run()
train:generate("hel", 10)

CONFIG (anytime after creation):
train:config{ lr = 0.01, stop_loss = 0.01 }
train:bar{ style = 2 }

DATA (pull from Hugging Face):
local data = ai.Data("wikitext/wikitext-2-v1", "500MB")
data:config{ batch_size = 8, tokenizer = tok }
data:batches()

CHECK GPU/BACKEND STATUS:
for _, b in ipairs(ai.Matrix.backends()) do print(b.name, b.available, b.reason) end
]=])
end

ai.h = ai.help

return ai