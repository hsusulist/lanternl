local Value = {}
Value.__index = Value

function Value.new(data, children, op)
    local self = setmetatable({}, Value)
    self.data = data
    self.grad = 0
    self.children = children or {}
    self.op = op or "leaf"
    self.backward_fn = function() end
    return self
end

Value.__add = function(a, b)
    if type(b) == "number" then b = Value.new(b) end
    local out = Value.new(a.data + b.data, {a, b}, "+")
    out.backward_fn = function()
        a.grad = a.grad + out.grad
        b.grad = b.grad + out.grad
    end
    return out
end

Value.__mul = function(a, b)
    if type(b) == "number" then b = Value.new(b) end
    local out = Value.new(a.data * b.data, {a, b}, "*")
    out.backward_fn = function()
        a.grad = a.grad + b.data * out.grad
        b.grad = b.grad + a.data * out.grad
    end
    return out
end


Value.__pow = function(a, n)
    local out = Value.new(a.data ^ n, {a}, "^" .. n)
    out.backward_fn = function()
        a.grad = a.grad + (n * a.data ^ (n - 1)) * out.grad
    end
    return out
end


function Value.relu(a)
    local out = Value.new(math.max(0, a.data), {a}, "relu")
    out.backward_fn = function()
        a.grad = a.grad + (a.data > 0 and 1 or 0) * out.grad
    end
    return out
end

function Value:backward()
    local topo = {}
    local visited = {}

    local function build(v)
        if not visited[v] then
            visited[v] = true
            for _, child in ipairs(v.children) do
                build(child)
            end
            table.insert(topo, v)
        end
    end
    build(self)

    self.grad = 1
    for i = #topo, 1, -1 do
        topo[i].backward_fn()
    end
end

Value.__tostring = function(v)
    return string.format("Value(data=%.4f, grad=%.4f)", v.data, v.grad)
end

return Value