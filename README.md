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

### 4. Train — train a model

```lua
local trainer = lanternl.Train{
    layers = {4, 8, 3},   -- architecture: input -> hidden -> output
    data = {
        {input = {1,0,0,0}, target = {1,0,0}},
        {input = {0,1,0,0}, target = {1,0,0}},
        {input = {0,0,1,0}, target = {0,1,0}},
        {input = {0,0,0,1}, target = {0,0,1}},
    },
    epochs = 200,
    lr = 0.05,
}

trainer:run()
trainer:evaluate()
```

Logging comes with a progress bar out of the box:
[####----------------] Epoch 40/200 | Loss: 0.04661 (best)


Want your own custom logging instead:
```lua
trainer:config{
    on_log = function(info)
        print("Epoch " .. info.epoch .. " loss=" .. info.loss)
    end
}
```
## Status

🚧 Work in progress — currently building Transformer components (attention, RMSNorm, RoPE) for full language model support.

## Author

Built by [hsusulist](https://github.com/hsusulist)
