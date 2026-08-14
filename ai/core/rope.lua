
local RoPE = {}
RoPE.__index = RoPE

local floor = math.floor
local cos, sin = math.cos, math.sin

local DEFAULT_BASE = 10000

function RoPE.new(config)
    config = config or {}
    local self = setmetatable({}, RoPE)

    self.head_dim = config.head_dim or config.dim
    if type(self.head_dim) ~= "number" or self.head_dim < 2 then
        error("rope: need config.head_dim >= 2, got " .. tostring(self.head_dim), 2)
    end
    self.head_dim = floor(self.head_dim)
    self.base = config.base or DEFAULT_BASE
    if type(self.base) ~= "number" or self.base <= 1 then
        error("rope: 'base' must be a number > 1, got " .. tostring(self.base), 2)
    end
    self.pairs = floor(self.head_dim / 2)

    self.inv_freq = {}
    for k = 1, self.pairs do
        self.inv_freq[k] = 1 / (self.base ^ ((2 * (k - 1)) / self.head_dim))
    end

    self.cos = {}   -- self.cos[pos + 1][k]
    self.sin = {}
    self.built = 0

    local pre = config.max_seq
    if type(pre) == "number" and pre > 0 then self:build(floor(pre)) end

    return self
end

-- Grow the cache so that positions 0 .. n-1 are available.
function RoPE:build(n)
    if n <= self.built then return self end
    local inv = self.inv_freq
    local npairs = self.pairs
    for pos = self.built, n - 1 do
        local c, s = {}, {}
        for k = 1, npairs do
            local angle = pos * inv[k]
            c[k] = cos(angle)
            s[k] = sin(angle)
        end
        self.cos[pos + 1] = c
        self.sin[pos + 1] = s
    end
    self.built = n
    return self
end

-- Returns two row-aligned tables: cos_rows[i] / sin_rows[i] correspond to
-- absolute position (i - 1 + offset). offset > 0 is for future KV-cache use.
function RoPE:rows(n, offset)
    offset = offset or 0
    self:build(n + offset)
    if offset == 0 then
        return self.cos, self.sin
    end
    local c, s = {}, {}
    for i = 1, n do
        c[i] = self.cos[i + offset]
        s[i] = self.sin[i + offset]
    end
    return c, s
end

return RoPE
