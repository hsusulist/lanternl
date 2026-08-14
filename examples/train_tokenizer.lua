local ai = require("ai")

-- 1. Train a BPE Tokenizer on some text
local tok = ai.Tokenizer.new {
    text = "hello world, this is a test of the lanternl tokenizer. it learns words and subwords.",
    vocab_size = 100,
    verbose = true
}
tok:train()

-- 2. Encode a string into token IDs
local text = "hello world"
local ids = tok:encode(text)

io.write("Encoded IDs:   ")
for _, id in ipairs(ids) do io.write(id, " ") end
print()

-- 3. Decode the IDs back into text
local decoded = tok:decode(ids)
print("Decoded text:  " .. decoded)

-- 4. Save the tokenizer for later use
tok:save("my_tokenizer.txt")
print("\nSaved tokenizer to 'my_tokenizer.txt'")



-- Heres a example of download a data
local dataset = ai.Data("your_username/your_dataset", "500MB")


-- You can pair it with tokenizer
local tok = ai.Tokenizer.new { text_file = dataset.files[1], vocab_size = 1000 }
tok:train()

dataset:config { batch_size = 8, tokenizer = tok }
local batches = dataset:batches()