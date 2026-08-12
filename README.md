#  lanternl

A deep learning library built from scratch in pure Lua — no Python, no external ML dependencies.

Inspired by PyTorch's simplicity, but made simpler — designed for embedding AI directly into Lua environments like Roblox, Love2D, and standalone Lua scripts.

## Why?

Most AI/ML tooling lives in Python. `lanternl` brings the same workflow — tokenizer, model, training loop — into Lua, so you can train and run small models anywhere Lua runs, without a Python runtime.

## Features

- **Autograd engine** — automatic differentiation from scratch (scalar + vectorized)
- **BPE Tokenizer** — trainable byte-pair encoding tokenizer
- **Full Transformer** — attention, RMSNorm, SwiGLU, GPU-accelerated (cuBLAS / custom CUDA kernels via FFI)
- **SGD optimizer** with auto learning-rate decay
- **Simple training loop** with default and custom logging
- **Dataset loader** — pull datasets from Hugging Face, batch, shuffle
- Clean, beginner-friendly config-based API

## Quick start

```lua
local ai = require("ai")
ai.help()   -- prints the full built-in guide, always up to date
```

That's it. `ai.help()` (or `ai.h()`) covers tokenizer, data loading, and training with real examples — no need to scroll through docs.

**The fastest way to train something right now:**

```lua
local train = ai.LMTrain{ data = "hello world this is my dataset", e = 200 }
train:run()
train:generate("hel", 10)
```

## Built-in for beginners

| Feature | What you get by default |
|---|---|
| Progress bar | `[####----] Epoch 40/200 \| Loss: 0.046 \| Acc: 62.3%` — no setup needed |
| Learning rate | Auto-decays over training, no tuning required |
| Stopping | Trains for `epochs`, or set `stop_loss = 0.01` to stop early |
| Data type | Pass plain text (`data = "..."`) — auto-tokenized, no manual encoding |
| Errors | Config mistakes (bad dims, missing data) raise clear messages, not silent failures |
| GPU | Auto-detected and used if available, falls back to CPU silently |

## For advanced use

Lanternl runs on pure Lua (CPU) out of the box, but you can train larger models and accelerate operations by pairing Lanternl with luaTL (LuaJIT + CUDA). luaTL is a small, dependency-free GPU math engine maintained at https://github.com/hsusulist/lua-TL and the repository includes a built-in adapter at `ai/core/luatl_adapter.lua`.

Minimal steps to use luaTL with Lanternl:

1. Build luaTL (requires NVIDIA CUDA Toolkit and LuaJIT):

```bash
# Linux / macOS
nvcc -O3 -shared -Xcompiler -fPIC -o libluaTL.so luaTL_core.cu

# Windows
nvcc -O3 -shared -o luaTL.dll luaTL_core.cu
```

2. Put the compiled `libluaTL.so` (or `luaTL.dll`) where LuaJIT can find it (project dir or system library path). Also include `luaTL.lua` alongside it or on `package.path`.

3. From your Lanternl project, ensure LuaJIT is used and require the adapter (Lanternl's `ai` automatically attempts to load `luatl_adapter`):

```lua
local ai = require("ai")
-- optional: force the luatl adapter so Lanternl uses GPU ops when available
ai.Matrix.use("luatl_adapter")

-- check available backends
for _, b in ipairs(ai.Matrix.backends()) do
  print(string.format("Backend: %-15s | Available: %-5s | Reason: %s",
        b.name, tostring(b.available), b.reason))
end
```

Notes and tips:
- luaTL requires LuaJIT (to use FFI) and a compatible CUDA driver/toolkit.
- If luaTL isn't present or fails to load, Lanternl will fall back to the pure-Lua CPU implementation automatically.
- For reproducible GPU usage call `ai.Matrix.use("luatl_adapter")` early in your script to force the backend.
- See https://github.com/hsusulist/lua-TL for full build instructions and examples.

Every default above is overridable:

```lua
train:config{ lr = 0.01, bar = 2, stop_loss = 0.005 }
train:config{ log = function(t) print(t.epoch, t.loss) end }
```

Run `ai.help()` anytime for the current full reference.


## Author

Built by [hsusulist](https://github.com/hsusulist)
