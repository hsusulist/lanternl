local Value = {}
Value.__index = Value

local exp, log, sqrt, huge = math.exp, math.log, math.sqrt, math.huge

local function wrap(x)
    if type(x) == "number" then return Value.new(x) end
    if type(x) == "table" and getmetatable(x) == Value then return x end
    error("Value: expected a number or a Value, got " .. type(x), 3)
end

function Value.new(data, children, op)
    if type(data) ~= "number" then
        error("Value.new: data must be a number, got " .. type(data), 2)
    end
    local self = setmetatable({}, Value)
    self.data = data
    self.grad = 0
    self.children = children or {}
    self.op = op or "leaf"
    self.backward_fn = false
    return self
end

function Value.is(v)
    return type(v) == "table" and getmetatable(v) == Value
end

Value.__add = function(a, b)
    a, b = wrap(a), wrap(b)
    local out = Value.new(a.data + b.data, { a, b }, "+")
    out.backward_fn = function()
        a.grad = a.grad + out.grad
        b.grad = b.grad + out.grad
    end
    return out
end

Value.__sub = function(a, b)
    a, b = wrap(a), wrap(b)
    local out = Value.new(a.data - b.data, { a, b }, "-")
    out.backward_fn = function()
        a.grad = a.grad + out.grad
        b.grad = b.grad - out.grad
    end
    return out
end

Value.__mul = function(a, b)
    a, b = wrap(a), wrap(b)
    local out = Value.new(a.data * b.data, { a, b }, "*")
    out.backward_fn = function()
        a.grad = a.grad + b.data * out.grad
        b.grad = b.grad + a.data * out.grad
    end
    return out
end

Value.__div = function(a, b)
    a, b = wrap(a), wrap(b)
    if b.data == 0 then
        error("Value./: division by zero", 2)
    end
    local out = Value.new(a.data / b.data, { a, b }, "/")
    out.backward_fn = function()
        a.grad = a.grad + (1 / b.data) * out.grad
        b.grad = b.grad - (a.data / (b.data * b.data)) * out.grad
    end
    return out
end

Value.__unm = function(a)
    a = wrap(a)
    local out = Value.new(-a.data, { a }, "neg")
    out.backward_fn = function()
        a.grad = a.grad - out.grad
    end
    return out
end

Value.__pow = function(a, b)
    if type(b) == "number" then
        a = wrap(a)
        if a.data < 0 and b ~= math.floor(b) then
            error("Value.^: negative base with a fractional exponent is not real", 2)
        end
        local out = Value.new(a.data ^ b, { a }, "^" .. b)
        out.backward_fn = function()
            a.grad = a.grad + (b * a.data ^ (b - 1)) * out.grad
        end
        return out
    end
    a, b = wrap(a), wrap(b)
    if a.data <= 0 then
        error("Value.^: base must be positive when the exponent is a Value", 2)
    end
    local val = a.data ^ b.data
    local out = Value.new(val, { a, b }, "^")
    out.backward_fn = function()
        a.grad = a.grad + b.data * a.data ^ (b.data - 1) * out.grad
        b.grad = b.grad + val * log(a.data) * out.grad
    end
    return out
end

Value.__lt = function(a, b) return wrap(a).data < wrap(b).data end
Value.__le = function(a, b) return wrap(a).data <= wrap(b).data end

function Value.relu(a)
    a = wrap(a)
    local out = Value.new(a.data > 0 and a.data or 0, { a }, "relu")
    out.backward_fn = function()
        a.grad = a.grad + (a.data > 0 and 1 or 0) * out.grad
    end
    return out
end

function Value:exp()
    local val = exp(self.data)
    local out = Value.new(val, { self }, "exp")
    out.backward_fn = function()
        self.grad = self.grad + val * out.grad
    end
    return out
end

function Value:log()
    if self.data <= 0 then
        error("Value:log: log is undefined for " .. tostring(self.data), 2)
    end
    local d = self.data
    local out = Value.new(log(d), { self }, "log")
    out.backward_fn = function()
        self.grad = self.grad + (1 / d) * out.grad
    end
    return out
end

function Value:sqrt()
    if self.data < 0 then
        error("Value:sqrt: sqrt is undefined for " .. tostring(self.data), 2)
    end
    local val = sqrt(self.data)
    local out = Value.new(val, { self }, "sqrt")
    out.backward_fn = function()
        if val ~= 0 then
            self.grad = self.grad + (0.5 / val) * out.grad
        end
    end
    return out
end

function Value:tanh()
    local d = self.data
    local t
    if d > 20 then t = 1
    elseif d < -20 then t = -1
    else
        local e = exp(2 * d)
        t = (e - 1) / (e + 1)
    end
    local out = Value.new(t, { self }, "tanh")
    out.backward_fn = function()
        self.grad = self.grad + (1 - t * t) * out.grad
    end
    return out
end

function Value:sigmoid()
    local d = self.data
    local s
    if d >= 0 then
        s = 1 / (1 + exp(-d))
    else
        local z = exp(d)
        s = z / (1 + z)
    end
    local out = Value.new(s, { self }, "sigmoid")
    out.backward_fn = function()
        self.grad = self.grad + s * (1 - s) * out.grad
    end
    return out
end

function Value:silu()
    local d = self.data
    local s
    if d >= 0 then
        s = 1 / (1 + exp(-d))
    else
        local z = exp(d)
        s = z / (1 + z)
    end
    local out = Value.new(d * s, { self }, "silu")
    out.backward_fn = function()
        self.grad = self.grad + s * (1 + d * (1 - s)) * out.grad
    end
    return out
end

function Value:item()
    return self.data
end

function Value:detach()
    return Value.new(self.data)
end

function Value:zero_grad()
    self.grad = 0
    return self
end

local function topo_sort(root)
    local topo, ntopo = {}, 0
    local visited = {}
    local stack_node, stack_i = { root }, { 1 }
    local sp = 1
    visited[root] = true
    while sp > 0 do
        local v = stack_node[sp]
        local idx = stack_i[sp]
        local child = v.children and v.children[idx] or nil
        if child ~= nil then
            stack_i[sp] = idx + 1
            if not visited[child] then
                visited[child] = true
                sp = sp + 1
                stack_node[sp] = child
                stack_i[sp] = 1
            end
        else
            ntopo = ntopo + 1
            topo[ntopo] = v
            stack_node[sp] = nil
            sp = sp - 1
        end
    end
    return topo, ntopo
end

function Value:zero_grad_graph()
    local topo, ntopo = topo_sort(self)
    for i = 1, ntopo do topo[i].grad = 0 end
    return self
end

function Value:backward(accumulate)
    local topo, ntopo = topo_sort(self)
    if not accumulate then
        for i = 1, ntopo do
            local v = topo[i]
            if v.op ~= "leaf" then v.grad = 0 end
        end
    end
    self.grad = 1
    for i = ntopo, 1, -1 do
        local f = topo[i].backward_fn
        if f then f() end
    end
    return self
end

Value.__tostring = function(v)
    return string.format("Value(data=%.6g, grad=%.6g)", v.data, v.grad)
end

return Value
