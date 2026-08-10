local Value = require("tensor")

local embedding = {}
embedding.__index = embedding

function embedding.new(config)
    local self = setmetatable({}, embedding)

    self.vocab_size = config.vocab_size
    self.dim = config.dim
    assert(self.vocab_size, "embedding: need config.vocab_size")
    assert(self.dim, "embedding: need config.dim")

    self.table = {}
    for id = 1, self.vocab_size do
        self.table[id] = {}
        for d = 1, self.dim do
            self.table[id][d] = Value.new(math.random() * 0.2 - 0.1)
        end
    end

    return self
end

function embedding:lookup(token_id)
    return self.table[token_id]
end

function embedding:lookup_all(token_ids)
    local out = {}
    for i, id in ipairs(token_ids) do
        out[i] = self:lookup(id)
    end
    return out
end

function embedding:parameters()
    local params = {}
    for id = 1, self.vocab_size do
        for d = 1, self.dim do
            table.insert(params, self.table[id][d])
        end
    end
    return params
end

function embedding:print(vec)
    for _, v in ipairs(vec) do
        io.write(string.format("%.3f ", v.data))
    end
    print()
end

return embedding