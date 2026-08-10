local positional = {}
positional.__index = positional

function positional.new(config)
    local self = setmetatable({}, positional)

    self.seq_len = config.seq_len
    self.dim = config.dim
    assert(self.seq_len, "positional: need config.seq_len")
    assert(self.dim, "positional: need config.dim")

    self.pe = {}
    for pos = 1, self.seq_len do
        self.pe[pos] = {}
        for i = 1, self.dim do
            local angle = pos / (10000 ^ ((i - 1) / self.dim))
            if i % 2 == 1 then
                self.pe[pos][i] = math.sin(angle)
            else
                self.pe[pos][i] = math.cos(angle)
            end
        end
    end

    return self
end

function positional:add(token_vecs)
    local out = {}
    for pos, vec in ipairs(token_vecs) do
        out[pos] = {}
        for d = 1, #vec do
            out[pos][d] = vec[d] + self.pe[pos][d]
        end
    end
    return out
end

return positional