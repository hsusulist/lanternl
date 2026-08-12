local matrix = require("matrix")
local ok_luatl, luatl_adapter = pcall(require, "luatl_adapter")
if not ok_luatl then luatl_adapter = { available = false } end

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
    local total = rows * cols
    for k = 1, total do
        m.data[k] = (math.random() * 2 - 1) * scale
    end
    return Tensor.new(m)
end

function Tensor.add(a, b)
    local out = Tensor.new(matrix.add(a.data, b.data), { a, b }, "+")
    out.backward_fn = function()
        local total = out.grad.rows * out.grad.cols
        for k = 1, total do
            a.grad.data[k] = a.grad.data[k] + out.grad.data[k]
            b.grad.data[k] = b.grad.data[k] + out.grad.data[k]
        end
    end
    return out
end

function Tensor.matmul(a, b)
    local out = Tensor.new(matrix.multiply(a.data, b.data), { a, b }, "matmul")
    out.backward_fn = function()
        local bt = matrix.transpose(b.data)
        local at = matrix.transpose(a.data)
        local da = matrix.multiply(out.grad, bt)
        local db = matrix.multiply(at, out.grad)

        local total_a = a.grad.rows * a.grad.cols
        for k = 1, total_a do
            a.grad.data[k] = a.grad.data[k] + da.data[k]
        end

        local total_b = b.grad.rows * b.grad.cols
        for k = 1, total_b do
            b.grad.data[k] = b.grad.data[k] + db.data[k]
        end
    end
    return out
end

function Tensor.relu(a)
    local out_data = matrix.new(a.data.rows, a.data.cols)
    local total = a.data.rows * a.data.cols
    local mask = {}

    for k = 1, total do
        local v = a.data.data[k]
        out_data.data[k] = v > 0 and v or 0
        mask[k] = v > 0 and 1 or 0
    end

    local out = Tensor.new(out_data, { a }, "relu")
    out.backward_fn = function()
        for k = 1, total do
            a.grad.data[k] = a.grad.data[k] + mask[k] * out.grad.data[k]
        end
    end
    return out
end

function Tensor.softmax_rows(a)
    local rows, cols = a.data.rows, a.data.cols
    local out_data = matrix.new(rows, cols)

    if luatl_adapter.available then
        out_data.data = luatl_adapter.softmax(a.data.data, rows, cols)
    else
        for i = 1, rows do
            local base = (i - 1) * cols
            local max_val = -math.huge
            for j = 1, cols do
                local v = a.data.data[base + j]
                if v > max_val then max_val = v end
            end

            local sum = 0
            for j = 1, cols do
                local e = math.exp(a.data.data[base + j] - max_val)
                out_data.data[base + j] = e
                sum = sum + e
            end

            for j = 1, cols do
                out_data.data[base + j] = out_data.data[base + j] / sum
            end
        end
    end

    local out = Tensor.new(out_data, { a }, "softmax")
    out.backward_fn = function()
        for i = 1, rows do
            local base = (i - 1) * cols
            local dot = 0
            for j = 1, cols do
                dot = dot + out.grad.data[base + j] * out_data.data[base + j]
            end
            for j = 1, cols do
                local y = out_data.data[base + j]
                a.grad.data[base + j] = a.grad.data[base + j] + y * (out.grad.data[base + j] - dot)
            end
        end
    end
    return out
end

function Tensor.slice_cols(a, start_col, end_col)
    local rows = a.data.rows
    local width = end_col - start_col + 1
    local out_data = matrix.new(rows, width)

    for i = 1, rows do
        local src_base = (i - 1) * a.data.cols
        local dst_base = (i - 1) * width
        for j = 1, width do
            out_data.data[dst_base + j] = a.data.data[src_base + start_col - 1 + j]
        end
    end

    local out = Tensor.new(out_data, { a }, "slice_cols")
    out.backward_fn = function()
        for i = 1, rows do
            local src_base = (i - 1) * a.grad.cols
            local dst_base = (i - 1) * width
            for j = 1, width do
                a.grad.data[src_base + start_col - 1 + j] =
                    a.grad.data[src_base + start_col - 1 + j] + out.grad.data[dst_base + j]
            end
        end
    end
    return out
end

function Tensor.concat_cols(tensors)
    local rows = tensors[1].data.rows
    local total_cols = 0
    for _, t in ipairs(tensors) do total_cols = total_cols + t.data.cols end

    local out_data = matrix.new(rows, total_cols)

    for i = 1, rows do
        local dst_base = (i - 1) * total_cols
        local col_offset = 0
        for _, t in ipairs(tensors) do
            local src_base = (i - 1) * t.data.cols
            for j = 1, t.data.cols do
                out_data.data[dst_base + col_offset + j] = t.data.data[src_base + j]
            end
            col_offset = col_offset + t.data.cols
        end
    end

    local out = Tensor.new(out_data, tensors, "concat_cols")
    out.backward_fn = function()
        for i = 1, rows do
            local dst_base = (i - 1) * total_cols
            local col_offset = 0
            for _, t in ipairs(tensors) do
                local src_base = (i - 1) * t.grad.cols
                for j = 1, t.grad.cols do
                    t.grad.data[src_base + j] = t.grad.data[src_base + j] + out.grad.data[dst_base + col_offset + j]
                end
                col_offset = col_offset + t.grad.cols
            end
        end
    end
    return out
end

function Tensor.silu(a)
    local total = a.data.rows * a.data.cols
    local out_data = matrix.new(a.data.rows, a.data.cols)
    local sig_values = {}

    for k = 1, total do
        local v = a.data.data[k]
        local sig = 1 / (1 + math.exp(-v))
        sig_values[k] = sig
        out_data.data[k] = v * sig
    end

    local out = Tensor.new(out_data, { a }, "silu")
    out.backward_fn = function()
        for k = 1, total do
            local v = a.data.data[k]
            local sig = sig_values[k]
            local dsilu = sig * (1 + v * (1 - sig))
            a.grad.data[k] = a.grad.data[k] + dsilu * out.grad.data[k]
        end
    end
    return out
end

function Tensor.mul(a, b)
    local total = a.data.rows * a.data.cols
    local out_data = matrix.new(a.data.rows, a.data.cols)
    for k = 1, total do
        out_data.data[k] = a.data.data[k] * b.data.data[k]
    end

    local out = Tensor.new(out_data, { a, b }, "mul")
    out.backward_fn = function()
        for k = 1, total do
            a.grad.data[k] = a.grad.data[k] + b.data.data[k] * out.grad.data[k]
            b.grad.data[k] = b.grad.data[k] + a.data.data[k] * out.grad.data[k]
        end
    end
    return out
end

function Tensor.cross_entropy(logits, target_ids)
    local rows, cols = logits.data.rows, logits.data.cols
    local softmax_vals = matrix.new(rows, cols)
    local total_loss = 0

    for i = 1, rows do
        local base = (i - 1) * cols
        local max_val = -math.huge
        for j = 1, cols do
            local v = logits.data.data[base + j]
            if v > max_val then max_val = v end
        end

        local sum = 0
        for j = 1, cols do
            local e = math.exp(logits.data.data[base + j] - max_val)
            softmax_vals.data[base + j] = e
            sum = sum + e
        end
        for j = 1, cols do
            softmax_vals.data[base + j] = softmax_vals.data[base + j] / sum
        end

        local target = target_ids[i]
        local p_target = softmax_vals.data[base + target]
        total_loss = total_loss - math.log(p_target + 1e-12)
    end

    local mean_loss = total_loss / rows

    local out_data = matrix.new(1, 1)
    out_data.data[1] = mean_loss
    local out = Tensor.new(out_data, { logits }, "cross_entropy")

    out.backward_fn = function()
        local g = out.grad.data[1] / rows
        for i = 1, rows do
            local base = (i - 1) * cols
            local target = target_ids[i]
            for j = 1, cols do
                local grad_val = softmax_vals.data[base + j]
                if j == target then grad_val = grad_val - 1 end
                logits.grad.data[base + j] = logits.grad.data[base + j] + grad_val * g
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

    local total = self.grad.rows * self.grad.cols
    for k = 1, total do
        self.grad.data[k] = 1
    end

    for i = #topo, 1, -1 do
        topo[i].backward_fn()
    end
end

function Tensor:zero_grad()
    local total = self.grad.rows * self.grad.cols
    for k = 1, total do
        self.grad.data[k] = 0
    end
end

function Tensor:print()
    matrix.print(self.data)
end

return Tensor
