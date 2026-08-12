# lanternl

A deep learning library built from scratch in pure Lua — no Python, no external ML dependencies. Inspired by PyTorch, designed for embedding AI into Lua environments (Roblox, Love2D, standalone Lua scripts).

## Stack

- **Language:** Lua / LuaJIT
- **GPU acceleration:** Optional — custom CUDA kernels (`luaTL`), cuBLAS via FFI, with automatic CPU fallback
- **No external ML dependencies** on CPU path

## Module layout

```
ai.lua                  Main entry point / public API
ai/core/
  matrix.lua            Row-major matrix type + backend registry (CPU / BLAS / CUDA)
  tensor.lua            Scalar autograd engine (legacy path)
  tensor2.lua           Matrix autograd engine (modern path)
  blas.lua              OpenBLAS FFI wrapper
  gpu.lua               CUDA / cuBLAS FFI wrapper
  luatl_adapter.lua     Custom luaTL CUDA kernel adapter
ai/nn/
  linear.lua            Linear layer (weight-only)
  attention.lua         Multi-head self-attention
  rmsnorm.lua           RMSNorm with trainable weight
  swiglu.lua            SwiGLU FFN block
  block.lua             Pre-norm Transformer block
  transformer.lua       Full Transformer model
  nn.lua                Legacy scalar MLP (unused by Transformer)
  embedding.lua         Legacy scalar embedding (unused)
  positional.lua        Legacy sinusoidal PE (unused)
ai/data/
  tokenizer.lua         BPE tokenizer (train / encode)
  data.lua              HuggingFace dataset loader, batching, shuffle
ai/optim/
  optim.lua             Legacy scalar SGD
  optim2.lua            Matrix Tensor SGD
  train.lua             Legacy scalar MLP trainer
  lmtrain.lua           Main LM trainer (config, training loop, generation, save/load, HF push)
test.lua                Export enumeration (minimal)
```

## Running on Replit

This is a **library**, not a web app. Install LuaJIT and run scripts directly:

```bash
luajit test.lua
```

GPU acceleration requires an NVIDIA GPU and compiled `luaTL.so` — not available on Replit. The CPU path works fine for small experiments.

## Known issues (discovered during import review)

- No causal attention mask — model is bidirectional despite being trained as autoregressive LM
- `Tokenizer` has no `decode()` or vocab→ID map; passing it to `Data` breaks `LMTrain`
- `zero_grad` called once per epoch, not per step — unintentional gradient accumulation
- CUDA wrappers ignore return codes — silent failure on GPU errors
- `lmtrain.lua` (655 lines) handles too many concerns; needs splitting
- Legacy modules (`nn.lua`, `embedding.lua`, `positional.lua`) are exported but unused

## User preferences

- User prefers the existing Lua structure to be preserved; no restructuring without asking
