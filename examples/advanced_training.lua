local ai = require("ai")

print("=== Advanced Training Example ===")

local trainer = ai.LMTrain {
    data = "the quick brown fox jumps over the lazy dog. the lazy dog sleeps all day.",
    preset = "auto",
    epochs = 1000,
    lr = 0.01,
    lr_decay = true,     -- Slow down learning over time
    stop_loss = 0.05,    -- Stop if loss hits 0.05
    patience = 50,       -- Stop if no improvement for 50 epochs
    verbose = false      -- We will use our own custom logger
}

-- Custom Logging (access internal state)
trainer:config {
    log = function(t)
        -- Only print every 100 epochs
        if t.epoch % 100 == 0 or t.epoch == 1 then
            print(string.format("Epoch %4d | Loss: %.4f | Acc: %.1f%%", t.epoch, t.loss, t.accuracy))
        end
    end
}

trainer:run()

print("\nTraining finished! Reason: " .. trainer.stopped)
print(string.format("Best Loss: %.4f", trainer.best_loss))

-- Generate with temperature for some randomness
print("\nGenerating text with temperature=0.5:")
local output = trainer:generate("the", 15, { temperature = 0.5 })
print(output)