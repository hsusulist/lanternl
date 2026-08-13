local matrix = require("matrix")

local Tensor = {}
local BACKWARD = {}

local floor = math.floor
local exp, log, sqrt, huge = math.exp, math.log, math.sqrt, math.huge
local sformat = string.format
local setmetatable, type = setmetatable, type

local alloc = matrix.alloc
local zeros = matrix.zeros

local grad_enabled = true
Tensor.training = false

local function grad_of(t)
    local g = t._grad
    if g == nil then
        g = zeros(t.data.rows, t.data.cols)
        t._grad = g
    end
    return g
end

Tensor.__index = function(t, k)
    if k == "grad" then return grad_of(t) end
    return Tensor[k]
end

Tensor.gradbuf = grad_of

local function node(data, children, op, ctx)
    return setmetatable({ data = data, children = children, op = op, ctx = ctx }, Tensor)
end

local function out_node(data, children, op, ctx)
    if not grad_enabled then return node(data, nil, "leaf", nil) end
    return node(data, children, op, ctx)
end

function Tensor.new(data_matrix, children, op, ctx)
    if type(data_matrix) ~= "table" or type(data_matrix.data) ~= "table" then
        error("Tensor.new: expected a matrix, got " .. type(data_matrix), 2)
    end
    return out_node(data_matrix, children, op or "leaf", ctx)
end

function Tensor.param(data_matrix)
    local t = node(data_matrix, nil, "leaf", nil)
    t._grad = zeros(data_matrix.rows, data_matrix.cols)
    t.is_param = true
    return t
end

function Tensor.is(t)
    return type(t) == "table" and getmetatable(t) == Tensor
end

function Tensor.set_grad_enabled(v)
    local prev = grad_enabled
    grad_enabled = v and true or false
    return prev
end

function Tensor.is_grad_enabled()
    return grad_enabled
end

function Tensor.no_grad(fn, ...)
    local prev = grad_enabled
    grad_enabled = false
    local ok, a, b, c = pcall(fn, ...)
    grad_enabled = prev
    if not ok then error(a, 2) end
    return a, b, c
end

function Tensor.from(t)
    return node(matrix.from(t), nil, "leaf", nil)
end

function Tensor.from_matrix(m)
    return node(m, nil, "leaf", nil)
end

function Tensor.zeros(rows, cols)
    return node(zeros(rows, cols), nil, "leaf", nil)
end

function Tensor.random(rows, cols, scale)
    scale = scale or 1
    local m = alloc(rows, cols)
    local d = m.data
    for k = 1, rows * cols do
        d[k] = (math.random() * 2 - 1) * scale
    end
    return node(m, nil, "leaf", nil)
end

local function same_shape(a, b, who)
    if a.data.rows ~= b.data.rows or a.data.cols ~= b.data.cols then
        error(sformat("%s: shape mismatch (%dx%d) vs (%dx%d)", who,
            a.data.rows, a.data.cols, b.data.rows, b.data.cols), 3)
    end
end

function Tensor.add(a, b)
    same_shape(a, b, "Tensor.add")
    return out_node(matrix.add(a.data, b.data), { a, b }, "add")
end

BACKWARD.add = function(nd)
    local g = nd._grad
    if not g then return end
    local kids = nd.children
    matrix.axpy(grad_of(kids[1]), 1, g)
    matrix.axpy(grad_of(kids[2]), 1, g)
end

function Tensor.sub(a, b)
    same_shape(a, b, "Tensor.sub")
    return out_node(matrix.sub(a.data, b.data), { a, b }, "sub")
end

BACKWARD.sub = function(nd)
    local g = nd._grad
    if not g then return end
    local kids = nd.children
    matrix.axpy(grad_of(kids[1]), 1, g)
    matrix.axpy(grad_of(kids[2]), -1, g)
end

function Tensor.add_rows(a, b)
    local rows, cols = a.data.rows, a.data.cols
    if b.data.rows ~= 1 or b.data.cols ~= cols then
        error(sformat("Tensor.add_rows: bias must be 1x%d, got %dx%d",
            cols, b.data.rows, b.data.cols), 2)
    end
    local out_data = alloc(rows, cols)
    local ad, bd, od = a.data.data, b.data.data, out_data.data
    for i = 1, rows do
        local base = (i - 1) * cols
        for j = 1, cols do
            od[base + j] = ad[base + j] + bd[j]
        end
    end
    return out_node(out_data, { a, b }, "add_rows", { rows = rows, cols = cols })
end

BACKWARD.add_rows = function(nd)
    local g = nd._grad
    if not g then return end
    local kids = nd.children
    local rows, cols = nd.ctx.rows, nd.ctx.cols
    matrix.axpy(grad_of(kids[1]), 1, g)
    local bg = grad_of(kids[2]).data
    local gd = g.data
    for i = 1, rows do
        local base = (i - 1) * cols
        for j = 1, cols do
            bg[j] = bg[j] + gd[base + j]
        end
    end
end

function Tensor.mul(a, b)
    same_shape(a, b, "Tensor.mul")
    return out_node(matrix.hadamard(a.data, b.data), { a, b }, "mul")
end

BACKWARD.mul = function(nd)
    local g = nd._grad
    if not g then return end
    local a, b = nd.children[1], nd.children[2]
    local ag, bg = grad_of(a).data, grad_of(b).data
    local ad, bd, gd = a.data.data, b.data.data, g.data
    for k = 1, g.rows * g.cols do
        local v = gd[k]
        ag[k] = ag[k] + bd[k] * v
        bg[k] = bg[k] + ad[k] * v
    end
end

function Tensor.scale(a, s)
    if type(s) ~= "number" then
        error("Tensor.scale: scalar must be a number, got " .. type(s), 2)
    end
    return out_node(matrix.scale(a.data, s), { a }, "scale", s)
end

BACKWARD.scale = function(nd)
    local g = nd._grad
    if not g then return end
    matrix.axpy(grad_of(nd.children[1]), nd.ctx, g)
end

function Tensor.matmul(a, b)
    if a.data.cols ~= b.data.rows then
        error(sformat("Tensor.matmul: shape mismatch (%dx%d) * (%dx%d)",
            a.data.rows, a.data.cols, b.data.rows, b.data.cols), 2)
    end
    return out_node(matrix.multiply(a.data, b.data), { a, b }, "matmul")
end

BACKWARD.matmul = function(nd)
    local g = nd._grad
    if not g then return end
    local a, b = nd.children[1], nd.children[2]
    matrix.matmul_nt(g, b.data, grad_of(a), true)
    matrix.matmul_tn(a.data, g, grad_of(b), true)
end

function Tensor.matmul_nt(a, b)
    if a.data.cols ~= b.data.cols then
        error(sformat("Tensor.matmul_nt: shape mismatch (%dx%d) * (%dx%d)^T",
            a.data.rows, a.data.cols, b.data.rows, b.data.cols), 2)
    end
    return out_node(matrix.matmul_nt(a.data, b.data), { a, b }, "matmul_nt")
end

BACKWARD.matmul_nt = function(nd)
    local g = nd._grad
    if not g then return end
    local a, b = nd.children[1], nd.children[2]
    matrix.multiply(g, b.data, grad_of(a), true)
    matrix.matmul_tn(g, a.data, grad_of(b), true)
end

function Tensor.transpose(a)
    return out_node(matrix.transpose(a.data), { a }, "transpose")
end

BACKWARD.transpose = function(nd)
    local g = nd._grad
    if not g then return end
    local ag = grad_of(nd.children[1])
    local rows, cols = g.rows, g.cols
    local gd, agd = g.data, ag.data
    for i = 1, rows do
        local base = (i - 1) * cols
        local idx = i
        for j = 1, cols do
            agd[idx] = agd[idx] + gd[base + j]
            idx = idx + rows
        end
    end
end

function Tensor.relu(a)
    local rows, cols = a.data.rows, a.data.cols
    local total = rows * cols
    local out_data = alloc(rows, cols)
    local src, dst = a.data.data, out_data.data
    for k = 1, total do
        local v = src[k]
        dst[k] = v > 0 and v or 0
    end
    return out_node(out_data, { a }, "relu")
end

BACKWARD.relu = function(nd)
    local g = nd._grad
    if not g then return end
    local ag = grad_of(nd.children[1]).data
    local od, gd = nd.data.data, g.data
    for k = 1, g.rows * g.cols do
        if od[k] > 0 then ag[k] = ag[k] + gd[k] end
    end
end

function Tensor.silu(a)
    local rows, cols = a.data.rows, a.data.cols
    local total = rows * cols
    local out_data = alloc(rows, cols)
    local src, dst = a.data.data, out_data.data
    local sig = grad_enabled and {} or nil
    for k = 1, total do
        local v = src[k]
        local s
        if v >= 0 then
            s = 1 / (1 + exp(-v))
        else
            local z = exp(v)
            s = z / (1 + z)
        end
        if sig then sig[k] = s end
        dst[k] = v * s
    end
    return out_node(out_data, { a }, "silu", sig)
end

BACKWARD.silu = function(nd)
    local g = nd._grad
    if not g then return end
    local a = nd.children[1]
    local ag, gd, ad = grad_of(a).data, g.data, a.data.data
    local sig = nd.ctx
    for k = 1, g.rows * g.cols do
        local s = sig[k]
        ag[k] = ag[k] + s * (1 + ad[k] * (1 - s)) * gd[k]
    end
end

function Tensor.tanh(a)
    local rows, cols = a.data.rows, a.data.cols
    local out_data = alloc(rows, cols)
    local src, dst = a.data.data, out_data.data
    for k = 1, rows * cols do
        local v = src[k]
        if v > 20 then dst[k] = 1
        elseif v < -20 then dst[k] = -1
        else
            local e = exp(2 * v)
            dst[k] = (e - 1) / (e + 1)
        end
    end
    return out_node(out_data, { a }, "tanh")
end

BACKWARD.tanh = function(nd)
    local g = nd._grad
    if not g then return end
    local ag = grad_of(nd.children[1]).data
    local od, gd = nd.data.data, g.data
    for k = 1, g.rows * g.cols do
        local t = od[k]
        ag[k] = ag[k] + (1 - t * t) * gd[k]
    end
end

function Tensor.softmax_rows(a, opts)
    local rows, cols = a.data.rows, a.data.cols
    local causal, offset = false, 0
    if opts == true then
        causal = true
    elseif type(opts) == "table" then
        causal = opts.causal and true or false
        offset = opts.offset or 0
    end

    local out_data = alloc(rows, cols)
    local src, dst = a.data.data, out_data.data

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
        if not (sum > 0) then sum = 1 end
        local inv = 1 / sum
        for j = 1, n do
            dst[base + j] = dst[base + j] * inv
        end
        for j = n + 1, cols do
            dst[base + j] = 0
        end
    end

    return out_node(out_data, { a }, "softmax",
        { rows = rows, cols = cols, limits = limits })
end

-- Softmax gradient: d/dx_i softmax(x) = softmax(x) * (delta_ij - softmax(x))
-- For numerical stability and efficiency, rearranged to: grad[i] = y[i] * (upstream_grad[i] - dot)
-- where dot = sum(upstream_grad * y). This avoids computing the full Jacobian matrix.
BACKWARD.softmax = function(nd)
    local g = nd._grad
    if not g then return end
    local ctx = nd.ctx
    local rows, cols, limits = ctx.rows, ctx.cols, ctx.limits
    local ag = grad_of(nd.children[1]).data
    local gd, yd = g.data, nd.data.data
    for i = 1, rows do
        local base = (i - 1) * cols
        local n = limits and limits[i] or cols
        local dot = 0
        for j = 1, n do
            dot = dot + gd[base + j] * yd[base + j]
        end
        for j = 1, n do
            local p = base + j
            local y = yd[p]
            ag[p] = ag[p] + y * (gd[p] - dot)
        end
    end
end

function Tensor.rope(a, cos_rows, sin_rows, npairs)
    local rows, cols = a.data.rows, a.data.cols
    npairs = npairs or floor(cols / 2)
    if 2 * npairs > cols then
        error("Tensor.rope: npairs*2 exceeds tensor width", 2)
    end
    for i = 1, rows do
        local c, s = cos_rows[i], sin_rows[i]
        if c == nil or s == nil then
            error("Tensor.rope: no cached angles for row " .. i, 2)
        end
        if #c < npairs or #s < npairs then
            error(sformat("Tensor.rope: row %d has %d angles but %d pairs are needed",
                i, #c, npairs), 2)
        end
    end

    local out_data = alloc(rows, cols)
    local src, dst = a.data.data, out_data.data
    for i = 1, rows do
        local base = (i - 1) * cols
        local c, s = cos_rows[i], sin_rows[i]
        for k = 1, npairs do
            local p = base + 2 * k - 1
            local x0, x1 = src[p], src[p + 1]
            local ck, sk = c[k], s[k]
            dst[p] = x0 * ck - x1 * sk
            dst[p + 1] = x0 * sk + x1 * ck
        end
        for j = 2 * npairs + 1, cols do
            dst[base + j] = src[base + j]
        end
    end

    return out_node(out_data, { a }, "rope",
        { cos = cos_rows, sin = sin_rows, npairs = npairs, rows = rows, cols = cols })
end

-- RoPE backward: Rotary position embedding applies 2D rotations to pairs of dimensions.
-- The gradient is the inverse rotation: d/dx (R(theta) * x) = R(-theta) * d/d(output)
-- This is computed as: [ag[p], ag[p+1]] = R(-theta) * [gd[p], gd[p+1]]
BACKWARD.rope = function(nd)
    local g = nd._grad
    if not g then return end
    local ctx = nd.ctx
    local rows, cols, npairs = ctx.rows, ctx.cols, ctx.npairs
    local cos_rows, sin_rows = ctx.cos, ctx.sin
    local ag = grad_of(nd.children[1]).data
    local gd = g.data
    for i = 1, rows do
        local base = (i - 1) * cols
        local c, s = cos_rows[i], sin_rows[i]
        for k = 1, npairs do
            local p = base + 2 * k - 1
            local g0, g1 = gd[p], gd[p + 1]
            local ck, sk = c[k], s[k]
            ag[p] = ag[p] + g0 * ck + g1 * sk
            ag[p + 1] = ag[p + 1] - g0 * sk + g1 * ck
        end
        for j = 2 * npairs + 1, cols do
            ag[base + j] = ag[base + j] + gd[base + j]
        end
    end
end

function Tensor.slice_cols(a, start_col, end_col)
    local rows, cols = a.data.rows, a.data.cols
    if not matrix.is_int(start_col) or not matrix.is_int(end_col) then
        error("Tensor.slice_cols: start_col and end_col must be integers", 2)
    end
    if start_col < 1 or end_col > cols or end_col < start_col then
        error(sformat("Tensor.slice_cols: range %d..%d is invalid for a %dx%d tensor",
            start_col, end_col, rows, cols), 2)
    end
    local width = end_col - start_col + 1
    local out_data = alloc(rows, width)
    local src, dst = a.data.data, out_data.data
    for i = 1, rows do
        local src_base = (i - 1) * cols + start_col - 1
        local dst_base = (i - 1) * width
        for j = 1, width do
            dst[dst_base + j] = src[src_base + j]
        end
    end
    return out_node(out_data, { a }, "slice_cols",
        { start_col = start_col, width = width, rows = rows, cols = cols })
end

BACKWARD.slice_cols = function(nd)
    local g = nd._grad
    if not g then return end
    local ctx = nd.ctx
    local rows, cols, width, start_col = ctx.rows, ctx.cols, ctx.width, ctx.start_col
    local ag = grad_of(nd.children[1]).data
    local gd = g.data
    for i = 1, rows do
        local src_base = (i - 1) * cols + start_col - 1
        local dst_base = (i - 1) * width
        for j = 1, width do
            local p = src_base + j
            ag[p] = ag[p] + gd[dst_base + j]
        end
    end
end

function Tensor.concat_cols(tensors)
    if type(tensors) ~= "table" or #tensors == 0 then
        error("Tensor.concat_cols: expected a non-empty list of tensors", 2)
    end
    local rows = tensors[1].data.rows
    local total_cols = 0
    for i = 1, #tensors do
        local t = tensors[i]
        if t.data.rows ~= rows then
            error(sformat("Tensor.concat_cols: tensor %d has %d rows but tensor 1 has %d",
                i, t.data.rows, rows), 2)
        end
        total_cols = total_cols + t.data.cols
    end

    local out_data = alloc(rows, total_cols)
    local dst = out_data.data
    for i = 1, rows do
        local dst_base = (i - 1) * total_cols
        local col_offset = 0
        for n = 1, #tensors do
            local t = tensors[n]
            local tc = t.data.cols
            local src_base = (i - 1) * tc
            local td = t.data.data
            for j = 1, tc do
                dst[dst_base + col_offset + j] = td[src_base + j]
            end
            col_offset = col_offset + tc
        end
    end
    return out_node(out_data, tensors, "concat_cols", { rows = rows, total = total_cols })
end

BACKWARD.concat_cols = function(nd)
    local g = nd._grad
    if not g then return end
    local rows, total_cols = nd.ctx.rows, nd.ctx.total
    local gd = g.data
    local kids = nd.children
    for i = 1, rows do
        local dst_base = (i - 1) * total_cols
        local col_offset = 0
        for n = 1, #kids do
            local t = kids[n]
            local tc = t.data.cols
            local tg = grad_of(t).data
            local src_base = (i - 1) * tc
            for j = 1, tc do
                local p = src_base + j
                tg[p] = tg[p] + gd[dst_base + col_offset + j]
            end
            col_offset = col_offset + tc
        end
    end
end

function Tensor.embed(weight, ids)
    local vocab, dim = weight.data.rows, weight.data.cols
    local n = #ids
    local out_data = alloc(n, dim)
    local wd, od = weight.data.data, out_data.data
    for i = 1, n do
        local id = ids[i]
        if not matrix.is_int(id) or id < 1 or id > vocab then
            error(sformat("Tensor.embed: token %d = %s is outside 1..%d",
                i, tostring(id), vocab), 2)
        end
        local src = (id - 1) * dim
        local dst = (i - 1) * dim
        for j = 1, dim do
            od[dst + j] = wd[src + j]
        end
    end
    return out_node(out_data, { weight }, "embed", { ids = ids, dim = dim, n = n })
end

BACKWARD.embed = function(nd)
    local g = nd._grad
    if not g then return end
    local ctx = nd.ctx
    local ids, dim, n = ctx.ids, ctx.dim, ctx.n
    local wg = grad_of(nd.children[1]).data
    local gd = g.data
    for i = 1, n do
        local dst = (ids[i] - 1) * dim
        local src = (i - 1) * dim
        for j = 1, dim do
            local p = dst + j
            wg[p] = wg[p] + gd[src + j]
        end
    end
end

function Tensor.rmsnorm(x, weight, eps)
    eps = eps or 1e-5
    local rows, cols = x.data.rows, x.data.cols
    if weight.data.rows ~= 1 or weight.data.cols ~= cols then
        error(sformat("Tensor.rmsnorm: weight must be 1x%d, got %dx%d",
            cols, weight.data.rows, weight.data.cols), 2)
    end
    local out_data = alloc(rows, cols)
    local xd, wd, od = x.data.data, weight.data.data, out_data.data
    local inv = {}
    for i = 1, rows do
        local base = (i - 1) * cols
        local ss = 0
        for j = 1, cols do
            local v = xd[base + j]
            ss = ss + v * v
        end
        local r = 1 / sqrt(ss / cols + eps)
        inv[i] = r
        for j = 1, cols do
            od[base + j] = xd[base + j] * r * wd[j]
        end
    end
    return out_node(out_data, { x, weight }, "rmsnorm",
        { inv = inv, rows = rows, cols = cols })
end

-- RMSNorm backward: d/dx RMSNorm(x, w) involves scaling through 1/sqrt(E[x^2] + eps)
-- Using the chain rule and simplifying, we get:
-- d(x)/dInput = (1/r) * d(output) * weight - (x/cols) * coef
-- where coef = (1/r)^3 * dot(d(output) * weight, x)
-- This form avoids computing the full Hessian and is numerically stable.
BACKWARD.rmsnorm = function(nd)
    local g = nd._grad
    if not g then return end
    local ctx = nd.ctx
    local rows, cols, inv = ctx.rows, ctx.cols, ctx.inv
    local x, w = nd.children[1], nd.children[2]
    local xg, wg = grad_of(x).data, grad_of(w).data
    local xd, wd, gd = x.data.data, w.data.data, g.data
    for i = 1, rows do
        local base = (i - 1) * cols
        local r = inv[i]
        local dot = 0
        for j = 1, cols do
            local p = base + j
            dot = dot + gd[p] * wd[j] * xd[p]
        end
        local coef = dot * r * r * r / cols
        for j = 1, cols do
            local p = base + j
            local xv = xd[p]
            local gv = gd[p]
            xg[p] = xg[p] + gv * wd[j] * r - xv * coef
            wg[j] = wg[j] + gv * xv * r
        end
    end
end

function Tensor.layernorm(x, weight, bias, eps)
    eps = eps or 1e-5
    local rows, cols = x.data.rows, x.data.cols
    local out_data = alloc(rows, cols)
    local xd, wd, od = x.data.data, weight.data.data, out_data.data
    local bd = bias and bias.data.data or nil
    local inv, mu = {}, {}
    for i = 1, rows do
        local base = (i - 1) * cols
        local s = 0
        for j = 1, cols do s = s + xd[base + j] end
        local m = s / cols
        local ss = 0
        for j = 1, cols do
            local d = xd[base + j] - m
            ss = ss + d * d
        end
        local r = 1 / sqrt(ss / cols + eps)
        mu[i], inv[i] = m, r
        for j = 1, cols do
            local nrm = (xd[base + j] - m) * r
            od[base + j] = nrm * wd[j] + (bd and bd[j] or 0)
        end
    end
    local kids = bias and { x, weight, bias } or { x, weight }
    return out_node(out_data, kids, "layernorm",
        { inv = inv, mu = mu, rows = rows, cols = cols })
end

BACKWARD.layernorm = function(nd)
    local g = nd._grad
    if not g then return end
    local ctx = nd.ctx
    local rows, cols, inv, mu = ctx.rows, ctx.cols, ctx.inv, ctx.mu
    local kids = nd.children
    local x, w, b = kids[1], kids[2], kids[3]
    local xg, wg = grad_of(x).data, grad_of(w).data
    local bg = b and grad_of(b).data or nil
    local xd, wd, gd = x.data.data, w.data.data, g.data
    for i = 1, rows do
        local base = (i - 1) * cols
        local m, r = mu[i], inv[i]
        local sum_dy, sum_dy_xhat = 0, 0
        for j = 1, cols do
            local p = base + j
            local dy = gd[p] * wd[j]
            local xhat = (xd[p] - m) * r
            sum_dy = sum_dy + dy
            sum_dy_xhat = sum_dy_xhat + dy * xhat
            wg[j] = wg[j] + gd[p] * xhat
            if bg then bg[j] = bg[j] + gd[p] end
        end
        for j = 1, cols do
            local p = base + j
            local dy = gd[p] * wd[j]
            local xhat = (xd[p] - m) * r
            xg[p] = xg[p] + r * (dy - sum_dy / cols - xhat * sum_dy_xhat / cols)
        end
    end
end

function Tensor.dropout(a, p)
    p = p or 0
    if p <= 0 or not Tensor.training or not grad_enabled then
        return a
    end
    if p >= 1 then
        error("Tensor.dropout: p must be < 1", 2)
    end
    local rows, cols = a.data.rows, a.data.cols
    local total = rows * cols
    local out_data = alloc(rows, cols)
    local src, dst = a.data.data, out_data.data
    local keep = 1 - p
    local invk = 1 / keep
    local mask = {}
    for k = 1, total do
        if math.random() < keep then
            mask[k] = invk
            dst[k] = src[k] * invk
        else
            mask[k] = 0
            dst[k] = 0
        end
    end
    return out_node(out_data, { a }, "dropout", mask)
end

BACKWARD.dropout = function(nd)
    local g = nd._grad
    if not g then return end
    local mask = nd.ctx
    local ag = grad_of(nd.children[1]).data
    local gd = g.data
    for k = 1, g.rows * g.cols do
        local m = mask[k]
        if m ~= 0 then ag[k] = ag[k] + m * gd[k] end
    end
end

function Tensor.sum(a)
    local total = a.data.rows * a.data.cols
    local ad = a.data.data
    local s = 0
    for k = 1, total do s = s + ad[k] end
    local out_data = alloc(1, 1)
    out_data.data[1] = s
    return out_node(out_data, { a }, "sum", total)
end

BACKWARD.sum = function(nd)
    local g = nd._grad
    if not g then return end
    local gv = g.data[1]
    local ag = grad_of(nd.children[1]).data
    for k = 1, nd.ctx do ag[k] = ag[k] + gv end
end

function Tensor.mean(a)
    local total = a.data.rows * a.data.cols
    local ad = a.data.data
    local s = 0
    for k = 1, total do s = s + ad[k] end
    local out_data = alloc(1, 1)
    out_data.data[1] = s / total
    return out_node(out_data, { a }, "mean", total)
end

BACKWARD.mean = function(nd)
    local g = nd._grad
    if not g then return end
    local total = nd.ctx
    local gv = g.data[1] / total
    local ag = grad_of(nd.children[1]).data
    for k = 1, total do ag[k] = ag[k] + gv end
end

function Tensor.detach(a)
    return node(matrix.copy(a.data), nil, "leaf", nil)
end

function Tensor.cross_entropy(logits, target_ids, opts)
    local rows, cols = logits.data.rows, logits.data.cols
    local ignore = opts and opts.ignore_index or nil
    local x = logits.data.data
    local sm = alloc(rows, cols)
    local smd = sm.data
    local total_loss = 0
    local counted = 0

    for i = 1, rows do
        local base = (i - 1) * cols
        local tgt = target_ids[i]
        if ignore ~= nil and tgt == ignore then
            for j = 1, cols do smd[base + j] = 0 end
        else
            if type(tgt) ~= "number" or tgt ~= floor(tgt) or tgt < 1 or tgt > cols then
                error(sformat(
                    "Tensor.cross_entropy: target %d is %s, expected an integer class id in 1..%d",
                    i, tostring(tgt), cols), 2)
            end
            local max_val = -huge
            for j = 1, cols do
                local v = x[base + j]
                if v > max_val then max_val = v end
            end
            if max_val == -huge or max_val ~= max_val then max_val = 0 end
            local sum = 0
            for j = 1, cols do
                local e = exp(x[base + j] - max_val)
                smd[base + j] = e
                sum = sum + e
            end
            local inv = 1 / sum
            for j = 1, cols do
                smd[base + j] = smd[base + j] * inv
            end
            total_loss = total_loss + ((max_val + log(sum)) - x[base + tgt])
            counted = counted + 1
        end
    end

    if counted == 0 then counted = 1 end
    local out_data = alloc(1, 1)
    out_data.data[1] = total_loss / counted

    return out_node(out_data, { logits }, "cross_entropy",
        { sm = smd, targets = target_ids, rows = rows, cols = cols,
          count = counted, ignore = ignore })
end

-- Cross-entropy backward: d/d(logits) CrossEntropy = softmax(logits) - one_hot(target)
-- This is the standard gradient formula from multi-class classification.
-- We compute this efficiently by: grad[i,j] = softmax[i,j] - (1 if j==target[i] else 0)
BACKWARD.cross_entropy = function(nd)
    local g = nd._grad
    if not g then return end
    local ctx = nd.ctx
    local rows, cols, smd, targets = ctx.rows, ctx.cols, ctx.sm, ctx.targets
    local ignore = ctx.ignore
    local gv = g.data[1] / ctx.count
    local lg = grad_of(nd.children[1]).data
    for i = 1, rows do
        local tgt = targets[i]
        if ignore == nil or tgt ~= ignore then
            local base = (i - 1) * cols
            for j = 1, cols do
                local p = base + j
                local gval = smd[p]
                if j == tgt then gval = gval - 1 end
                lg[p] = lg[p] + gval * gv
            end
        end
    end
end

function Tensor:item()
    return self.data.data[1]
end

function Tensor:backward(accumulate)
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
            if not accumulate and v.children and v._grad then
                matrix.zero(v._grad)
            end
            stack_node[sp] = nil
            sp = sp - 1
        end
    end

    local seed = grad_of(self)
    local sd = seed.data
    for k = 1, seed.rows * seed.cols do sd[k] = 1 end

    for i = ntopo, 1, -1 do
        local v = topo[i]
        if v._grad then
            local f = BACKWARD[v.op]
            if f then
                f(v)
            elseif v.backward_fn then
                v.backward_fn()
            end
        end
    end
    return self
end

function Tensor:zero_grad()
    if self._grad then matrix.zero(self._grad) end
    return self
end

function Tensor:set_grad(m)
    self._grad = m
    return self
end

function Tensor:print(fmt)
    matrix.print(self.data, fmt)
    return self
end

Tensor.ops = BACKWARD

return Tensor
