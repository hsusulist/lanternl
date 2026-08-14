local ai = require("ai")

print("=== LanternL Quickstart ===")
print("Training a model on 'hello world'...")

-- Pass a string, auto-size the AI, train, and generate!
local model = ai.LMTrain {
    data = "hello world, this is lua ai!",
    preset = "auto",   -- Automatically picks model dimensions
    epochs = 300,
    verbose = false    -- Hide the progress bar for this quick example
}

model:run()

-- Give it a seed, and it generates text!
local output = model:generate("hello", 10)
print("\nGenerated text:")
print(output)