# 🏮 lanternl

A deep learning library built from scratch in pure Lua — no Python, no external ML dependencies.

Inspired by PyTorch's simplicity but i made it simpler , designed for embedding AI directly into Lua environments like Roblox, Love2D, and standalone Lua scripts.

## Why?

Most AI/ML tooling lives in Python. `lanternl` brings the same workflow — tokenizer, model, training loop — into Lua, so you can train and run small models anywhere Lua runs, without a Python runtime.

## Features

- **Autograd engine** — automatic differentiation from scratch (scalar + vectorized)
- **BPE Tokenizer** — trainable byte-pair encoding tokenizer
- **Neural network layers** — Neuron, Layer, MLP with configurable architecture
- **SGD optimizer**
- **Simple training loop** with default and custom logging
- **Dataset loader** — pull datasets from Hugging Face, batch, shuffle
- Clean, beginner-friendly config-based API

## Quick start

### 1. Require the library

```lua
local lanternl = require("lanternl")
```

That's the only line you need — everything else (`Tokenizer`, `Data`, `Train`, `Model`...) lives inside `lanternl`.

### 2. Tokenizer — turn text into tokens

```lua
local tok = lanternl.Tokenizer.new{
    text = "hello world this is my dataset",
    vocab_size = 60,
}

tok:train()

local ids = tok:encode("hello world")
tok:print(ids)   -- [he] [l] [l] [o] [</w>] [w] [or] [l] [d] [</w>]

tok:save("merges.txt")   -- save so you don't have to retrain later
```

To train on a larger text file instead of typing text directly:

```lua
local tok = lanternl.Tokenizer.new{
    text_file = "my_dataset.txt",
    vocab_size = 2000,
}
```

### 3. Data — pull a dataset (from Hugging Face)

```lua
local data = lanternl.Data("wikitext/wikitext-2-v1", "500MB")

data:config{
    batch_size = 8,
    shuffle = true,
    tokenizer = tok,   -- auto-encodes using the tokenizer trained in step 2
}

local batches = data:batches()
```

`"500MB"` caps how much gets downloaded — swap in `"1GB"`, `"100 pairs"`, etc. to avoid pulling a huge dataset by accident.

### 4. Train — Ultra-Flexible Training API

**Option A: The Lazy Way (Instant Setup)**
```lua
local ai = require("lanternl")

local train = ai.LMTrain{
    e = 200,   -- 'e' or 'epochs'
    data = {2, 5, 8, 3, 9, 1, 4, 7, 6, 10},
}

train:run()
Option B: Model Config + Advanced Early Stopping

Lua
local train = ai.LMTrain{
    vocab = 30, dim = 16, layers = 2, heads = 4, ffn = 32,
    e = 500,
    data = {2, 5, 8, 3, 9, 1, 4, 7, 6, 10},
    stop_loss = 0.01,  -- Auto-stop when loss hits target
}

-- Custom Progress Bar style (e.g. ▓▓▓░░░)
train:config{ bar = 2 }

-- Custom Logger using live state (t.loss, t.epoch)
train:config{
    log = function(t)
        print("Epoch: " .. t.epoch .. " | Loss: " .. t.loss)
    end
}
## Status

🚧 Work in progress — currently testing and upgrading

## Author

Built by [hsusulist](https://github.com/hsusulist)
