local matrix = require("matrix")
local ok_luatl, luatl_adapter = pcall(require, "luatl_adapter")
if not ok_luatl then luatl_adapter = { available = false } end

local Tensor = {}
Tensor.__index = Tensor

local floor = math.floor
local exp, log, huge = math.exp, math.log, math.huge

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
        local og, ag, bg = out.grad.data, a.grad.data, b.grad.data
        for k = 1, total do
            local g = og[k]
            ag[k] = ag[k] + g
            bg[k] = bg[k] + g
        end
    end
    return out
end

function Tensor.matmul(a, b)
    if a.data.cols ~= b.data.rows then
        error(string.format("Tensor.matmul: shape mismatch (%dx%d) * (%dx%d)",
            a.data.rows, a.data.cols, b.data.rows, b.data.cols), 2)
    end
    local out = Tensor.new(matrix.multiply(a.data, b.data), { a, b }, "matmul")
    out.backward_fn = function()
        local bt = matrix.transpose(b.data)
        local at = matrix.transpose(a.data)
        local da = matrix.multiply(out.grad, bt)
        local db = matrix.multiply(at, out.grad)

        local ag, dad = a.grad.data, da.data
        local total_a = a.grad.rows * a.grad.cols
        for k = 1, total_a do
            ag[k] = ag[k] + dad[k]
        end

        local bg, dbd = b.grad.data, db.data
        local total_b = b.grad.rows * b.grad.cols
        for k = 1, total_b do
            bg[k] = bg[k] + dbd[k]
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
        local ag, og = a.grad.data, out.grad.data
        for k = 1, total do
            ag[k] = ag[k] + mask[k] * og[k]
        end
    end
    return out
end

--------------------------------------------------------------------------
-- softmax over rows, with optional causal (autoregressive) masking.
--
--   Tensor.softmax_rows(x)                        -- dense, unchanged
--   Tensor.softmax_rows(x, true)                  -- row i sees cols 1..i
--   Tensor.softmax_rows(x, { causal = true,
--                            offset = k })        -- row i sees cols 1..i+k
--
-- Masked entries are exact zeros, produced by simply not visiting them --
-- no -inf / -1e30 sentinel is added to the scores. That matters because
-- softmax_rows subtracts the row max: a sentinel would either participate
-- in the max scan or risk inf-inf = NaN in a float32 GPU kernel. Skipping
-- is also ~2x less work.
--
-- Backward is exact: for a masked column y_j = 0, so
--   dL/dx_j = y_j * (g_j - sum_a g_a y_a) = 0
-- for every masked j. The skipped grad slots therefore already hold the
-- correct analytic value (zero), and no gradient can leak from a future
-- position back into q/k at an earlier position.
--------------------------------------------------------------------------
function Tensor.softmax_rows(a, opts)
    local rows, cols = a.data.rows, a.data.cols

    local causal, offset = false, 0
    if opts == true then
        causal = true
    elseif type(opts) == "table" then
        causal = opts.causal and true or false
        offset = opts.offset or 0
    end

    local out_data = matrix.new(rows, cols)
    local src = a.data.data
    local dst = out_data.data

    -- limits[i] = number of visible columns in row i (nil => dense)
    local limits = nil
    if causal then
        limits = {}
        for i = 1, rows do
            local lim = i + offset
            if lim > cols then lim = cols end
            if lim < 1 then lim = 1 end
            limits[i] = lim
        end
    end

    local used_adapter = false
    if luatl_adapter.available then
        if causal and type(luatl_adapter.softmax_causal) == "function" then
            -- expected kernel signature: (flat, rows, cols, offset) -> flat
            out_data.data = luatl_adapter.softmax_causal(src, rows, cols, offset)
            dst = out_data.data
            used_adapter = true
        elseif (not causal) and type(luatl_adapter.softmax) == "function" then
            out_data.data = luatl_adapter.softmax(src, rows, cols)
            dst = out_data.data
            used_adapter = true
        end
    end

    if not used_adapter then
        for i = 1, rows do
            local base = (i - 1) * cols
            local n = limits and limits[i] or cols

            local max_val = -huge
            for j = 1, n do
                local v = src[base + j]
                if v > max_val then max_val = v end
            end
            if max_val == -huge or max_val ~= max_val then max_val = 0 end

            local sum = 0
            for j = 1, n do
                local e = exp(src[base + j] - max_val)
                dst[base + j] = e
                sum = sum + e
            end
            if not (sum > 0) then sum = 1 end   -- unreachable for causal (diagonal always visible)

            local inv = 1 / sum
            for j = 1, n do
                dst[base + j] = dst[base + j] * inv
            end
            for j = n + 1, cols do
                dst[base + j] = 0
            end
        end
    end

    local out = Tensor.new(out_data, { a }, causal and "softmax_causal" or "softmax")
    out.backward_fn = function()
        local og, ag = out.grad.data, a.grad.data
        for i = 1, rows do
            local base = (i - 1) * cols
            local n = limits and limits[i] or cols
            local dot = 0
            for j = 1, n do
                dot = dot + og[base + j] * dst[base + j]
            end
            for j = 1, n do
                local y = dst[base + j]
                ag[base + j] = ag[base + j] + y * (og[base + j] - dot)
            end
            -- columns n+1..cols: dL/dx = 0 exactly, nothing to accumulate.
        end
    end
    return out
end

--------------------------------------------------------------------------
-- Rotary position embedding, applied per row (row i => position i-1+offset).
-- cos_rows/sin_rows come from rope.lua (:rows(n, offset)).
-- The rotation is orthogonal, so backward is the transposed rotation.
--------------------------------------------------------------------------
function Tensor.rope(a, cos_rows, sin_rows, npairs)
    local rows, cols = a.data.rows, a.data.cols
    npairs = npairs or floor(cols / 2)
    if 2 * npairs > cols then
        error("Tensor.rope: npairs*2 exceeds tensor width", 2)
    end

    local out_data = matrix.new(rows, cols)
    local src, dst = a.data.data, out_data.data

    for i = 1, rows do
        local base = (i - 1) * cols
        local c, s = cos_rows[i], sin_rows[i]
        if c == nil then
            error("Tensor.rope: no cached angles for row " .. i, 2)
        end
        for k = 1, npairs do
            local p = base + 2 * k - 1
            local x0, x1 = src[p], src[p + 1]
            local ck, sk = c[k], s[k]
            dst[p]     = x0 * ck - x1 * sk
            dst[p + 1] = x0 * sk + x1 * ck
        end
        for j = 2 * npairs + 1, cols do
            dst[base + j] = src[base + j]
        end
    end

    local out = Tensor.new(out_data, { a }, "rope")
    out.backward_fn = function()
        local og, ag = out.grad.data, a.grad.data
        for i = 1, rows do
            local base = (i - 1) * cols
            local c, s = cos_rows[i], sin_rows[i]
            for k = 1, npairs do
                local p = base + 2 * k - 1
                local g0, g1 = og[p], og[p + 1]
                local ck, sk = c[k], s[k]
                ag[p]     = ag[p]     + g0 * ck + g1 * sk
                ag[p + 1] = ag[p + 1] - g0 * sk + g1 * ck
            end
            for j = 2 * npairs + 1, cols do
                ag[base + j] = ag[base + j] + og[base + j]
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
        local ag, og = a.grad.data, out.grad.data
        for i = 1, rows do
            local src_base = (i - 1) * a.grad.cols
            local dst_base = (i - 1) * width
            for j = 1, width do
                local p = src_base + start_col - 1 + j
                ag[p] = ag[p] + og[dst_base + j]
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
        local og = out.grad.data
        for i = 1, rows do
            local dst_base = (i - 1) * total_cols
            local col_offset = 0
            for _, t in ipairs(tensors) do
                local src_base = (i - 1) * t.grad.cols
                local tg = t.grad.data
                for j = 1, t.grad.cols do
                    tg[src_base + j] = tg[src_base + j] + og[dst_base + col_offset + j]
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
        -- exp(-v) overflows for very negative v; clamp keeps this finite.
        local sig
        if v >= 0 then
            sig = 1 / (1 + exp(-v))
        else
            local z = exp(v)
            sig = z / (1 + z)
        end
        sig_values[k] = sig
        out_data.data[k] = v * sig
    end

    local out = Tensor.new(out_data, { a }, "silu")
    out.backward_fn = function()
        local ag, og, ad = a.grad.data, out.grad.data, a.data.data
        for k = 1, total do
            local v = ad[k]
            local sig = sig_values[k]
            local dsilu = sig * (1 + v * (1 - sig))
            ag[k] = ag[k] + dsilu * og[k]
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
        local ag, bg, og = a.grad.data, b.grad.data, out.grad.data
        local ad, bd = a.data.data, b.data.data
        for k = 1, total do
            local g = og[k]
            ag[k] = ag[k] + bd[k] * g
            bg[k] = bg[k] + ad[k] * g
        end
    end
    return out
end

-- Mean cross-entropy over rows, computed as logsumexp(x) - x[target].
-- Exact log-softmax: no "+1e-12" epsilon, so a genuinely confident-wrong
-- prediction reports its real loss instead of saturating at ~27.6.
function Tensor.cross_entropy(logits, target_ids)
    local rows, cols = logits.data.rows, logits.data.cols
    local x = logits.data.data
    local softmax_vals = matrix.new(rows, cols)
    local sm = softmax_vals.data
    local total_loss = 0

    for i = 1, rows do
        local target = target_ids[i]
        if type(target) ~= "number" or target ~= floor(target) or target < 1 or target > cols then
            error(string.format(
                "Tensor.cross_entropy: target %d is %s, expected an integer class id in 1..%d",
                i, tostring(target), cols), 2)
        end

        local base = (i - 1) * cols
        local max_val = -huge
        for j = 1, cols do
            local v = x[base + j]
            if v > max_val then max_val = v end
        end
        if max_val == -huge or max_val ~= max_val then max_val = 0 end

        local sum = 0
        for j = 1, cols do
            local e = exp(x[base + j] - max_val)
            sm[base + j] = e
            sum = sum + e
        end

        local inv = 1 / sum
        for j = 1, cols do
            sm[base + j] = sm[base + j] * inv
        end

        -- -log p_target  ==  logsumexp(x) - x_target
        total_loss = total_loss + ((max_val + log(sum)) - x[base + target])
    end

    local mean_loss = total_loss / rows

    local out_data = matrix.new(1, 1)
    out_data.data[1] = mean_loss
    local out = Tensor.new(out_data, { logits }, "cross_entropy")

    out.backward_fn = function()
        local g = out.grad.data[1] / rows
        local lg = logits.grad.data
        for i = 1, rows do
            local base = (i - 1) * cols
            local target = target_ids[i]
            for j = 1, cols do
                local grad_val = sm[base + j]
                if j == target then grad_val = grad_val - 1 end
                lg[base + j] = lg[base + j] + grad_val * g
            end
        end
    end

    return out
end

-- Iterative post-order DFS: same ordering as the old recursive build(),
-- but the graph depth is now bounded by heap, not by the Lua C stack.
function Tensor:backward()
    local topo, ntopo = {}, 0
    local visited = {}
    local stack_node, stack_i = { self }, { 1 }
    local sp = 1
    visited[self] = true

    while sp > 0 do
        local v = stack_node[sp]
        local idx = stack_i[sp]
        local kids = v.children
        local child = kids and kids[idx] or nil
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

    local total = self.grad.rows * self.grad.cols
    for k = 1, total do
        self.grad.data[k] = 1
    end

    for i = ntopo, 1, -1 do
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
