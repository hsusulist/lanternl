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
    Train       = "train",
}

for upper_name, file_name in pairs(modules) do
    local lib = safe_require(file_name)
    ai[upper_name] = lib
    ai[file_name]  = lib
end

return ai