local tokenizer = {}
tokenizer.__index = tokenizer

local sformat = string.format
local sbyte, schar, ssub, sfind = string.byte, string.char, string.sub, string.find
local concat = table.concat

local FORMAT_VERSION = 2

local function pretokenize(s, out)
    out = out or {}
    local i, n = 1, #s
    while i <= n do
        local st, en = sfind(s, "^ ?%a+", i)
        if not st then st, en = sfind(s, "^ ?%d+", i) end
        if not st then st, en = sfind(s, "^ ?[^%s%w]+", i) end
        if not st then st, en = sfind(s, "^%s+", i) end
        if not st then st, en = i, i end
        out[#out + 1] = ssub(s, st, en)
        i = en + 1
    end
    return out
end

local ESCAPES = { ["\\"] = "\\\\", ["\n"] = "\\n", ["\r"] = "\\r", ["\t"] = "\\t" }

local function escape(s)
    return (s:gsub("[%c\\]", function(c)
        local e = ESCAPES[c]
        if e then return e end
        return sformat("\\x%02X", sbyte(c))
    end))
end

local function unescape(s)
    local out = {}
    local i, n = 1, #s
    while i <= n do
        local c = ssub(s, i, i)
        if c == "\\" then
            local nx = ssub(s, i + 1, i + 1)
            if nx == "n" then out[#out + 1] = "\n"; i = i + 2
            elseif nx == "r" then out[#out + 1] = "\r"; i = i + 2
            elseif nx == "t" then out[#out + 1] = "\t"; i = i + 2
            elseif nx == "\\" then out[#out + 1] = "\\"; i = i + 2
            elseif nx == "x" then
                local hex = ssub(s, i + 2, i + 3)
                local b = tonumber(hex, 16)
                if not b then error("tokenizer: bad \\x escape in merge file", 2) end
                out[#out + 1] = schar(b)
                i = i + 4
            else
                error("tokenizer: unknown escape '\\" .. nx .. "' in merge file", 2)
            end
        else
            out[#out + 1] = c
            i = i + 1
        end
    end
    return concat(out)
end

local Heap = {}
Heap.__index = Heap

local function heap_key(a, b)
    return sformat("%d\1%s%s", #a, a, b)
end

local function better(x, y)
    if x.c ~= y.c then return x.c > y.c end
    return x.key < y.key
end

function Heap.new()
    return setmetatable({ n = 0 }, Heap)
end

function Heap:push(c, a, b)
    local e = { c = c, a = a, b = b, key = heap_key(a, b) }
    local n = self.n + 1
    self.n = n
    self[n] = e
    while n > 1 do
        local parent = math.floor(n / 2)
        if better(self[n], self[parent]) then
            self[n], self[parent] = self[parent], self[n]
            n = parent
        else
            break
        end
    end
end

function Heap:pop()
    local n = self.n
    if n == 0 then return nil end
    local top = self[1]
    self[1] = self[n]
    self[n] = nil
    self.n = n - 1
    n = n - 1
    local i = 1
    while true do
        local l, r = 2 * i, 2 * i + 1
        local best = i
        if l <= n and better(self[l], self[best]) then best = l end
        if r <= n and better(self[r], self[best]) then best = r end
        if best == i then break end
        self[i], self[best] = self[best], self[i]
        i = best
    end
    return top
end

local function rebuild_vocab(self)
    self.vocab = {}
    self.inv_vocab = {}
    self.ranks = {}

    local function add_token(t)
        if self.vocab[t] == nil then
            self.inv_vocab[#self.inv_vocab + 1] = t
            self.vocab[t] = #self.inv_vocab
        end
    end

    for _, t in ipairs(self.special_tokens or {}) do add_token(t) end
    for i = 0, 255 do add_token(schar(i)) end
    for i = 1, #self.merges do
        local m = self.merges[i]
        add_token(m[1] .. m[2])
        local ra = self.ranks[m[1]]
        if ra == nil then ra = {}; self.ranks[m[1]] = ra end
        if ra[m[2]] == nil then ra[m[2]] = i end
    end

    self.vocab_size = #self.inv_vocab
    self.unk_id = self.vocab["<unk>"]
    self.memo = {}
    self.trained = true
    return self
end

function tokenizer.new(config)
    config = config or {}
    local self = setmetatable({}, tokenizer)

    self.text = config.text
    self.text_file = config.text_file
    self.vocab_size = config.vocab_size or 500
    self.min_frequency = config.min_frequency or 2
    self.special_tokens = config.special_tokens or { "<pad>", "<s>", "</s>", "<unk>" }
    self.log_every = config.log_every or 50
    self.verbose = (config.verbose == nil) and true or config.verbose

    self.merges = {}
    self.trained = false
    self.memo = {}

    return self
end

function tokenizer:log(msg)
    if self.verbose then print(msg) end
end

function tokenizer:target_merges()
    local reserved = #self.special_tokens + 256
    local n = self.vocab_size - reserved
    if n < 0 then n = 0 end
    return n
end

local function count_words(text, freqs)
    local chunks = pretokenize(text)
    for i = 1, #chunks do
        local c = chunks[i]
        freqs[c] = (freqs[c] or 0) + 1
    end
    return freqs
end

function tokenizer:_collect()
    local freqs = {}
    if self.text then
        count_words(self.text, freqs)
    elseif self.text_file then
        local f = io.open(self.text_file, "rb")
        if not f then
            error("tokenizer: cannot open text_file '" .. tostring(self.text_file) .. "'", 3)
        end
        local carry = ""
        while true do
            local block = f:read(1024 * 256)
            if not block then break end
            block = carry .. block
            local cut = #block
            while cut > 0 and not sfind(ssub(block, cut, cut), "%s") do
                cut = cut - 1
            end
            if cut == 0 then
                carry = block
            else
                count_words(ssub(block, 1, cut), freqs)
                carry = ssub(block, cut + 1)
            end
        end
        if #carry > 0 then count_words(carry, freqs) end
        f:close()
    else
        error("tokenizer: need config.text or config.text_file", 3)
    end
    return freqs
end

function tokenizer:train()
    self.merges = {}
    local freqs = self:_collect()

    local words, wfreq, nwords = {}, {}, 0
    for w, c in pairs(freqs) do
        nwords = nwords + 1
        local toks = {}
        for i = 1, #w do toks[i] = ssub(w, i, i) end
        words[nwords] = toks
        wfreq[nwords] = c
    end
    self:log(sformat("Unique chunks: %d", nwords))

    local counts, where = {}, {}

    local function bump(a, b, delta, widx)
        local ca = counts[a]
        if ca == nil then ca = {}; counts[a] = ca end
        local cur = (ca[b] or 0) + delta
        if cur <= 0 then
            ca[b] = nil
            local wa = where[a]
            if wa and wa[b] then wa[b] = nil end
        else
            ca[b] = cur
            local wa = where[a]
            if wa == nil then wa = {}; where[a] = wa end
            local wb = wa[b]
            if wb == nil then wb = {}; wa[b] = wb end
            if widx then wb[widx] = true end
        end
        return cur
    end

    for i = 1, nwords do
        local t = words[i]
        local fq = wfreq[i]
        for j = 1, #t - 1 do
            bump(t[j], t[j + 1], fq, i)
        end
    end

    local heap = Heap.new()
    for a, row in pairs(counts) do
        for b, c in pairs(row) do
            heap:push(c, a, b)
        end
    end

    local want = self:target_merges()
    local min_freq = self.min_frequency

    while #self.merges < want do
        local top
        while true do
            top = heap:pop()
            if top == nil then break end
            local row = counts[top.a]
            local cur = row and row[top.b]
            if cur ~= nil and cur == top.c then break end
        end
        if top == nil then
            self:log("No more pairs, stopping early.")
            break
        end
        if top.c < min_freq then
            self:log(sformat("Best pair count %d is below min_frequency %d, stopping.",
                top.c, min_freq))
            break
        end

        local a, b = top.a, top.b
        local merged = a .. b
        self.merges[#self.merges + 1] = { a, b }

        local wa = where[a]
        local affected = wa and wa[b] or {}
        local touched = {}
        for widx in pairs(affected) do touched[#touched + 1] = widx end

        for n = 1, #touched do
            local widx = touched[n]
            local t = words[widx]
            local fq = wfreq[widx]
            local len = #t
            local hit = false
            for j = 1, len - 1 do
                if t[j] == a and t[j + 1] == b then hit = true break end
            end
            if hit then
                for j = 1, len - 1 do
                    bump(t[j], t[j + 1], -fq, nil)
                end
                local nt, ni, j = {}, 0, 1
                while j <= len do
                    if j < len and t[j] == a and t[j + 1] == b then
                        ni = ni + 1
                        nt[ni] = merged
                        j = j + 2
                    else
                        ni = ni + 1
                        nt[ni] = t[j]
                        j = j + 1
                    end
                end
                words[widx] = nt
                for j2 = 1, ni - 1 do
                    local x, y = nt[j2], nt[j2 + 1]
                    local cur = bump(x, y, fq, widx)
                    heap:push(cur, x, y)
                end
            end
        end

        local ca = counts[a]
        if ca then ca[b] = nil end
        local wa2 = where[a]
        if wa2 then wa2[b] = nil end

        if #self.merges % self.log_every == 0 then
            self:log(sformat("Merge #%d: %q + %q (count=%d)", #self.merges, a, b, top.c))
        end
    end

    rebuild_vocab(self)
    self:log(sformat("Vocabulary: %d tokens from %d merges", self.vocab_size, #self.merges))
    return self
end

function tokenizer:_bpe(chunk)
    local memo = self.memo
    local cached = memo[chunk]
    if cached then return cached end

    local toks = {}
    for i = 1, #chunk do toks[i] = ssub(chunk, i, i) end

    local ranks = self.ranks
    while #toks > 1 do
        local best_rank, best_i = nil, nil
        for i = 1, #toks - 1 do
            local row = ranks[toks[i]]
            local r = row and row[toks[i + 1]]
            if r and (best_rank == nil or r < best_rank) then
                best_rank, best_i = r, i
            end
        end
        if best_i == nil then break end
        local merged = toks[best_i] .. toks[best_i + 1]
        table.remove(toks, best_i + 1)
        toks[best_i] = merged
    end

    memo[chunk] = toks
    return toks
end

function tokenizer:encode(text)
    if not self.trained then
        error("tokenizer:encode: tokenizer has no vocabulary yet, call :train() or :load() first", 2)
    end
    if type(text) ~= "string" then
        error("tokenizer:encode: expected a string, got " .. type(text), 2)
    end
    local result, n = {}, 0
    local chunks = pretokenize(text)
    local vocab, unk = self.vocab, self.unk_id
    for i = 1, #chunks do
        local toks = self:_bpe(chunks[i])
        for j = 1, #toks do
            local id = vocab[toks[j]] or unk
            if id == nil then
                error("tokenizer:encode: token " .. sformat("%q", toks[j])
                    .. " is not in the vocabulary and there is no <unk>", 2)
            end
            n = n + 1
            result[n] = id
        end
    end
    return result
end

function tokenizer:decode(tokens)
    if not self.trained then
        error("tokenizer:decode: tokenizer has no vocabulary yet, call :train() or :load() first", 2)
    end
    local specials = {}
    for _, t in ipairs(self.special_tokens or {}) do specials[t] = true end
    local out, n = {}, 0
    for i = 1, #tokens do
        local t = self.inv_vocab[tokens[i]]
        if t and not specials[t] then
            n = n + 1
            out[n] = t
        end
    end
    return concat(out)
end

function tokenizer:save(path)
    local f, err = io.open(path, "wb")
    if not f then
        error("tokenizer:save: cannot write '" .. tostring(path) .. "': " .. tostring(err), 2)
    end
    f:write(sformat("bpe\t%d\n", FORMAT_VERSION))
    f:write(sformat("specials\t%d\n", #self.special_tokens))
    for _, t in ipairs(self.special_tokens) do
        f:write(escape(t), "\n")
    end
    f:write(sformat("merges\t%d\n", #self.merges))
    for _, m in ipairs(self.merges) do
        f:write(escape(m[1]), "\t", escape(m[2]), "\n")
    end
    f:close()
    self:log("Saved tokenizer to " .. path)
    return self
end

function tokenizer:load(path)
    local f, err = io.open(path, "rb")
    if not f then
        error("tokenizer:load: cannot open '" .. tostring(path) .. "': " .. tostring(err), 2)
    end
    local lines = {}
    for line in f:lines() do lines[#lines + 1] = line end
    f:close()

    local li = 1
    local function next_line()
        local l = lines[li]
        li = li + 1
        return l
    end

    local header = next_line()
    local kind, version = (header or ""):match("^(%a+)\t(%d+)$")
    if kind ~= "bpe" then
        error("tokenizer:load: '" .. path .. "' is not a versioned bpe file "
            .. "(re-train and :save() to upgrade the old lossy format)", 2)
    end
    version = tonumber(version)
    if version > FORMAT_VERSION then
        error(sformat("tokenizer:load: file format v%d is newer than supported v%d",
            version, FORMAT_VERSION), 2)
    end

    local nspec = tonumber((next_line() or ""):match("^specials\t(%d+)$") or "")
    if not nspec then error("tokenizer:load: missing specials header", 2) end
    local specials = {}
    for i = 1, nspec do
        specials[i] = unescape(next_line() or "")
    end

    local nmerge = tonumber((next_line() or ""):match("^merges\t(%d+)$") or "")
    if not nmerge then error("tokenizer:load: missing merges header", 2) end
    local merges = {}
    for i = 1, nmerge do
        local line = next_line()
        if not line then
            error(sformat("tokenizer:load: file ends after %d of %d merges", i - 1, nmerge), 2)
        end
        local a, b = line:match("^(.-)\t(.*)$")
        if not a then
            error(sformat("tokenizer:load: malformed merge on line %d", li - 1), 2)
        end
        merges[i] = { unescape(a), unescape(b) }
    end

    self.special_tokens = specials
    self.merges = merges
    rebuild_vocab(self)
    self:log(sformat("Loaded tokenizer from %s (%d merges, %d tokens)",
        path, #merges, self.vocab_size))
    return self
end

function tokenizer:print(tokens)
    for _, t in ipairs(tokens) do
        io.write("[", tostring(self.inv_vocab and self.inv_vocab[t] or t), "] ")
    end
    print()
end

tokenizer.pretokenize = pretokenize

return tokenizer
