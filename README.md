# LANTERNL

A deep learning library built from scratch in pure Lua — no Python, no external ML dependencies.

Inspired by PyTorch's simplicity, but made simpler — designed for embedding AI directly into Lua environments like Roblox, Love2D, and standalone Lua scripts.

## Why?

Most AI/ML tooling lives in Python. `lanternl` brings the same workflow — tokenizer, model, training loop — into Lua, so you can train and run models anywhere Lua runs, without a Python runtime.

## Features

- **Autograd engine** — automatic differentiation from scratch (scalar + vectorized)
- **BPE Tokenizer** — trainable byte-pair encoding tokenizer
- **Full Transformer** — attention, RMSNorm, SwiGLU
- **GPU acceleration** — custom CUDA kernels (luaTL) and cuBLAS via FFI, with automatic CPU fallback
- **SGD optimizer** with auto learning-rate decay
- **Training loop** with default and custom logging, early stopping, save/load, HuggingFace push
- **Dataset loader** — pull datasets from Hugging Face, batch, shuffle
- Clean, beginner-friendly config-based API

## Quick start

```lua
local ai = require("ai")
ai.help()   -- prints the full built-in guide, always up to date
```

**Train something right now:**

```lua
local train = ai.LMTrain{ data = "hello world this is my dataset", e = 200 }
train:run()
train:generate("hel", 10)
```

This runs on CPU with pure Lua loops. Fine for small experiments. For anything bigger, you want GPU acceleration — see below.

## Training larger models (GPU / Colab)

Your local machine (Windows, etc.) runs plain Lua or LuaJIT on CPU only. To train larger models at real speed, you need a GPU — Google Colab provides one for free (Tesla T4).

**1. Open a new Colab notebook, enable a GPU:**

Runtime → Change runtime type → select "T4 GPU" → Save.

**2. Set up the environment (run once per notebook):**

```python
!apt-get install -y luajit -qq
!git clone https://github.com/hsusulist/lanternl.git
%cd lanternl
!nvcc -O3 -shared -Xcompiler -fPIC -o luaTL.so ai/core/luaTL_core.cu
```

**3. Verify GPU acceleration is active:**

```python
%%script luajit
package.path = package.path .. ";./ai/core/?.lua;./ai/nn/?.lua;./ai/data/?.lua;./ai/optim/?.lua"
local ai = require("ai")
for _, b in ipairs(ai.Matrix.backends()) do
    print(b.name, b.available, b.reason)
end
```

`luatl_adapter` should show `true, ready`. That's the custom CUDA backend — it's tried first, before cuBLAS, before CPU.

**4. Train a larger model:**

```python
%%script luajit
package.path = package.path .. ";./ai/core/?.lua;./ai/nn/?.lua;./ai/data/?.lua;./ai/optim/?.lua"
local ai = require("ai")

local train = ai.LMTrain{
    vocab = 2000, dim = 128, layers = 4, heads = 8, ffn = 256,
    e = 1000,
    data = "your larger training text goes here...",
}
train:run()
train:save("model.txt")
```

Scale `vocab`, `dim`, `layers`, `ffn` up as far as your GPU's memory (T4 = 16GB) allows. Small matrix multiplies fall back to CPU automatically (GPU round-trip overhead isn't worth it below a size threshold) — this is expected and handled for you.

**5. Save your work before the Colab session ends:**

```lua
train:save("model.txt")           -- local file
train:push("your-username/repo")  -- upload to Hugging Face (requires huggingface-cli login first)
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

Every default above is overridable:

```lua
train:config{ lr = 0.01, bar = 2, stop_loss = 0.005 }
train:config{ log = function(t) print(t.epoch, t.loss) end }
```

Run `ai.help()` anytime for the current full reference.

## Status

Work in progress — currently testing and upgrading.

## Author

Built by [hsusulist](https://github.com/hsusulist)
