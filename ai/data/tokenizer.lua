local tokenizer = {}
tokenizer.__index = tokenizer

local function to_chars(word)
    local chars = {}
    for i = 1, #word do
        table.insert(chars, word:sub(i, i))
    end
    return chars
end

local function split_key(key)
    local t = {}
    for tok in key:gmatch("%S+") do table.insert(t, tok) end
    return t
end

-- Rebuild the integer vocabulary from the merge list. This is shared by
-- train() and load(), since a loaded tokenizer does not run through training.
local function rebuild_vocab(self)
    self.vocab = {}
    self.inv_vocab = {}

    local function add_token(t)
        if not self.vocab[t] then
            self.vocab[t] = #self.inv_vocab + 1
            table.insert(self.inv_vocab, t)
        end
    end

    for _, t in ipairs(self.special_tokens or {}) do add_token(t) end
    for i = 0, 255 do add_token(string.char(i)) end
    add_token("</w>")
    for _, m in ipairs(self.merges) do
        add_token(m[1] .. m[2])
    end

    self.vocab_size = #self.inv_vocab
end

function tokenizer.new(config)
    config = config or {}
    local self = setmetatable({}, tokenizer)

    self.text = config.text
    self.text_file = config.text_file
    self.vocab_size = config.vocab_size or 500
    self.min_frequency = config.min_frequency or 2
    self.special_tokens = config.special_tokens or {"<pad>", "<s>", "</s>", "<unk>"}
    self.log_every = config.log_every or 50
    self.verbose = (config.verbose == nil) and true or config.verbose

    self.merges = {}

    return self
end

function tokenizer:log(msg)
    if self.verbose then print(msg) end
end

function tokenizer:train()
    self.merges = {}
    local text = self.text
    if not text and self.text_file then
        local f = io.open(self.text_file, "r")
        if not f then
            error("tokenizer: cannot open text_file '" .. self.text_file .. "'", 2)
        end
        text = f:read("*a")
        f:close()
    end
    assert(text, "tokenizer: need config.text or config.text_file")

    local word_freqs = {}
    for word in text:gmatch("%S+") do
        local chars = to_chars(word)
        table.insert(chars, "</w>")
        local key = table.concat(chars, " ")
        word_freqs[key] = (word_freqs[key] or 0) + 1
    end

    local unique_count = 0
    for _ in pairs(word_freqs) do unique_count = unique_count + 1 end
    self:log(string.format("Unique words: %d", unique_count))

    while true do
        local vocab = {}
        for key, _ in pairs(word_freqs) do
            for _, tok in ipairs(split_key(key)) do
                vocab[tok] = true
            end
        end
        local vocab_count = 0
        for _ in pairs(vocab) do vocab_count = vocab_count + 1 end

        if vocab_count >= self.vocab_size then
            self:log(string.format("Reached vocab size: %d", vocab_count))
            break
        end

        local pair_counts = {}
        for key, freq in pairs(word_freqs) do
            local tokens = split_key(key)
            for i = 1, #tokens - 1 do
                local pair_key = tokens[i] .. "\1" .. tokens[i+1]
                pair_counts[pair_key] = (pair_counts[pair_key] or 0) + freq
            end
        end

        local best_pair, best_count = nil, 0
        for pair_key, count in pairs(pair_counts) do
            if count > best_count then
                best_count = count
                best_pair = pair_key
            end
        end

        if not best_pair or best_count < self.min_frequency then
            self:log("No more useful merges, stopping early.")
            break
        end

        local a, b = best_pair:match("^(.-)\1(.+)$")
        table.insert(self.merges, {a, b})

        local new_word_freqs = {}
        for key, freq in pairs(word_freqs) do
            local tokens = split_key(key)
            local new_tokens = {}
            local i = 1
            while i <= #tokens do
                if i < #tokens and tokens[i] == a and tokens[i+1] == b then
                    table.insert(new_tokens, a .. b)
                    i = i + 2
                else
                    table.insert(new_tokens, tokens[i])
                    i = i + 1
                end
            end
            local new_key = table.concat(new_tokens, " ")
            new_word_freqs[new_key] = (new_word_freqs[new_key] or 0) + freq
        end
        word_freqs = new_word_freqs

        if #self.merges % self.log_every == 0 then
            self:log(string.format("Merge #%d: '%s' + '%s' (count=%d)", #self.merges, a, b, best_count))
        end
    end

    rebuild_vocab(self)
    return self
end

function tokenizer:encode(text)
    local result = {}
    for word in text:gmatch("%S+") do
        local tokens = to_chars(word)
        table.insert(tokens, "</w>")

        for _, merge in ipairs(self.merges) do
            local a, b = merge[1], merge[2]
            local new_tokens = {}
            local i = 1
            while i <= #tokens do
                if i < #tokens and tokens[i] == a and tokens[i+1] == b then
                    table.insert(new_tokens, a .. b)
                    i = i + 2
                else
                    table.insert(new_tokens, tokens[i])
                    i = i + 1
                end
            end
            tokens = new_tokens
        end

        for _, t in ipairs(tokens) do
            table.insert(result, self.vocab[t] or self.vocab["<unk>"])
        end
    end
    return result
end

function tokenizer:decode(tokens)
    local text = {}
    for _, id in ipairs(tokens) do
        local t = self.inv_vocab[id]
        if t then
            local cleaned = t:gsub("</w>", " ")
            table.insert(text, cleaned)
        end
    end
    local result = table.concat(text)
    return result:gsub("%s+", " "):match("^%s*(.-)%s*$")
end

function tokenizer:save(path)
    local f = io.open(path, "w")
    for _, m in ipairs(self.merges) do
        f:write(m[1] .. "\t" .. m[2] .. "\n")
    end
    f:close()
    self:log("Saved to " .. path)
end

function tokenizer:load(path)
    local f, err = io.open(path, "r")
    if not f then
        error("tokenizer: cannot open '" .. tostring(path) .. "': " .. tostring(err), 2)
    end

    self.merges = {}
    for line in f:lines() do
        local a, b = line:match("^(.-)\t(.+)$")
        if not a or not b then
            f:close()
            error("tokenizer: malformed merge line in '" .. tostring(path) .. "'", 2)
        end
        table.insert(self.merges, { a, b })
    end
    f:close()

    rebuild_vocab(self)
    self:log("Loaded from " .. path)
    return self
end

function tokenizer:print(tokens)
    for _, t in ipairs(tokens) do
        io.write("[" .. t .. "] ")
    end
    print()
end

return tokenizer