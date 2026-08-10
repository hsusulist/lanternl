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

```lua
local lanternl = require("lanternl")

-- Tokenizer
local tok = lanternl.Tokenizer.new{
    text = "hello world this is my dataset",
    vocab_size = 60,
}
tok:train()
local ids = tok:encode("hello world")
tok:print(ids)

-- Training
local trainer = lanternl.Train{
    layers = {4, 8, 3},
    data = {
        {input = {1,0,0,0}, target = {1,0,0}},
        {input = {0,1,0,0}, target = {1,0,0}},
        {input = {0,0,1,0}, target = {0,1,0}},
        {input = {0,0,0,1}, target = {0,0,1}},
    }
}
trainer:run()
trainer:evaluate()
```

## Project structure
