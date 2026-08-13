-- positional.lua -- fixed sinusoidal absolute position encoding.
-- Kept for the older non-Transformer code path and selectable in
-- transformer.lua via pos = "sinusoidal". The Transformer default is RoPE
-- (rope.lua), which is relative, parameter-free and length-extrapolating.
--
-- Fixed vs the previous version:
--   * positions are now 0-based (pos 0 => sin 0 / cos 1), as in the paper
--   * a sin/cos PAIR now shares one frequency: exponent 2*floor((i-1)/2)/dim
--     instead of (i-1)/dim. Without this, PE(pos+k) is not a fixed linear
--     map of PE(pos) and the encoding loses its relative-offset property.
--   * the table grows lazily, so :add() past seq_len no longer nil-indexes.

local ok_matrix, matrix = pcall(require, "matrix")

local positional = {}
positional.__index = positional

local floor = math.floor
local sin, cos = math.sin, math.cos

local DEFAULT_BASE = 10000

function positional.new(config)
    config = config or {}
    local self = setmetatable({}, positional)

    self.dim = config.dim
    assert(type(self.dim) == "number" and self.dim >= 1, "positional: need config.dim")
    self.dim = floor(self.dim)
    self.base = config.base or DEFAULT_BASE

    self.inv_freq = {}
    for i = 1, self.dim do
        local pair = floor((i - 1) / 2)
        self.inv_freq[i] = 1 / (self.base ^ ((2 * pair) / self.dim))
    end

    self.pe = {}
    self.built = 0
    self.seq_len = config.seq_len or 0
    if self.seq_len > 0 then self:build(self.seq_len) end

    return self
end

-- make positions 0 .. n-1 available (pe[pos+1])
function positional:build(n)
    if n <= self.built then return self end
    local dim, inv = self.dim, self.inv_freq
    for pos = self.built, n - 1 do
        local row = {}
        for i = 1, dim do
            local angle = pos * inv[i]
            if i % 2 == 1 then
                row[i] = sin(angle)
            else
                row[i] = cos(angle)
            end
        end
        self.pe[pos + 1] = row
    end
    self.built = n
    if n > self.seq_len then self.seq_len = n end
    return self
end

-- rows[i] is the encoding for position i-1
function positional:rows(n)
    self:build(n)
    return self.pe
end

-- legacy API: token_vecs is an array of per-position vectors
function positional:add(token_vecs)
    local n = #token_vecs
    self:build(n)
    local out = {}
    for pos = 1, n do
        local vec = token_vecs[pos]
        local row = self.pe[pos]
        local dst = {}
        local width = #vec
        for d = 1, width do
            dst[d] = vec[d] + (row[d] or 0)
        end
        out[pos] = dst
    end
    return out
end

-- flat matrix.lua helpers (in-place add, and a standalone PE matrix)
function positional:add_matrix(m, scale)
    scale = scale or 1
    assert(m.cols == self.dim, "positional: matrix width must equal dim")
    self:build(m.rows)
    for i = 1, m.rows do
        local base = (i - 1) * m.cols
        local row = self.pe[i]
        for j = 1, m.cols do
            m.data[base + j] = m.data[base + j] + row[j] * scale
        end
    end
    return m
end

function positional:matrix(n)
    assert(ok_matrix, "positional:matrix requires matrix.lua")
    self:build(n)
    local m = matrix.new(n, self.dim)
    for i = 1, n do
        local base = (i - 1) * self.dim
        local row = self.pe[i]
        for j = 1, self.dim do
            m.data[base + j] = row[j]
        end
    end
    return m
end

return positional
