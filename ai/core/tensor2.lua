local matrix = require("matrix")

local Tensor = {}
Tensor.__index = Tensor

function Tensor.new(data_matrix, children, op)
    local self = setmetatable({}, Tensor)
    self.data = data_matrix
    self.grad = matrix.new(data_matrix.rows, data_matrix.cols)
    self.children = children or {}
    self.op = op or "leaf"
    self.backward_fn = function() end
    return self
end

function Tensor.from(t)
    return Tensor.new(matrix.from(t))
end

function Tensor.zeros(rows, cols)
    return Tensor.new(matrix.new(rows, cols))
end

function Tensor.random(rows, cols, scale)
    scale = scale or 1
    local m = matrix.new(rows, cols)
    for i = 1, rows do
        for j = 1, cols do
            m.data[i][j] = (math.random() * 2 - 1) * scale
        end
    end
    return Tensor.new(m)
end

function Tensor.add(a, b)
    local out = Tensor.new(matrix.add(a.data, b.data), {a, b}, "+")
    out.backward_fn = function()
        for i = 1, out.grad.rows do
            for j = 1, out.grad.cols do
                a.grad.data[i][j] = a.grad.data[i][j] + out.grad.data[i][j]
                b.grad.data[i][j] = b.grad.data[i][j] + out.grad.data[i][j]
            end
        end
    end
    return out
end

function Tensor.matmul(a, b)
    local out = Tensor.new(matrix.multiply(a.data, b.data), {a, b}, "matmul")
    out.backward_fn = function()
        local bt = matrix.transpose(b.data)
        local at = matrix.transpose(a.data)
        local da = matrix.multiply(out.grad, bt)
        local db = matrix.multiply(at, out.grad)
        for i = 1, a.grad.rows do
            for j = 1, a.grad.cols do
                a.grad.data[i][j] = a.grad.data[i][j] + da.data[i][j]
            end
        end
        for i = 1, b.grad.rows do
            for j = 1, b.grad.cols do
                b.grad.data[i][j] = b.grad.data[i][j] + db.data[i][j]
            end
        end
    end
    return out
end

function Tensor.relu(a)
    local out_data = matrix.new(a.data.rows, a.data.cols)
    local mask = matrix.new(a.data.rows, a.data.cols)
    for i = 1, a.data.rows do
        for j = 1, a.data.cols do
            local v = a.data.data[i][j]
            out_data.data[i][j] = math.max(0, v)
            mask.data[i][j] = v > 0 and 1 or 0
        end
    end
    local out = Tensor.new(out_data, {a}, "relu")
    out.backward_fn = function()
        for i = 1, a.grad.rows do
            for j = 1, a.grad.cols do
                a.grad.data[i][j] = a.grad.data[i][j] + mask.data[i][j] * out.grad.data[i][j]
            end
        end
    end
    return out
end

function Tensor:backward()
    local topo = {}
    local visited = {}

    local function build(v)
        if not visited[v] then
            visited[v] = true
            for _, c in ipairs(v.children) do build(c) end
            table.insert(topo, v)
        end
    end
    build(self)

    for i = 1, self.grad.rows do
        for j = 1, self.grad.cols do
            self.grad.data[i][j] = 1
        end
    end

    for i = #topo, 1, -1 do
        topo[i].backward_fn()
    end
end

function Tensor:zero_grad()
    for i = 1, self.grad.rows do
        for j = 1, self.grad.cols do
            self.grad.data[i][j] = 0
        end
    end
end

function Tensor:print()
    matrix.print(self.data)
end

return Tensor