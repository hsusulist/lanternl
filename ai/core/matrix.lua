local matrix = {}

local floor = math.floor
local huge = math.huge
local sformat = string.format
local concat = table.concat

matrix.accel = { enabled = true, threshold = 4096, max_errors = 3, force = nil }
matrix.last_backend = "lua"

local BACKEND_NAMES = { "luatl_adapter", "gpu", "blas" }
local backends = {}

for i = 1, #BACKEND_NAMES do
    local name = BACKEND_NAMES[i]
    local entry = { name = name, available = false, errors = 0, reason = "not loaded" }
    local ok, mod = pcall(require, name)
    if not ok then
        local msg = tostring(mod)
        msg = msg:match("^[^\n]*") or msg
        entry.reason = "not installed (" .. msg .. ")"
    elseif type(mod) ~= "table" then
        entry.reason = "module did not return a table (got " .. type(mod) .. ")"
    else
        entry.module = mod
        if mod.available ~= true then
            entry.reason = "module reports available = " .. tostring(mod.available)
        elseif type(mod.matmul) ~= "function" then
            entry.reason = "module is available but exposes no matmul() function"
        else
            entry.available = true
            entry.reason = "ready"
        end
    end
    backends[i] = entry
end

local function is_int(x)
    return type(x) == "number" and x == floor(x) and x == x and x ~= huge and x ~= -huge
end

local function check_matrix(m, who, argname)
    if type(m) ~= "table" then
        error(sformat("matrix.%s: %s must be a matrix table, got %s", who, argname, type(m)), 3)
    end
    if not is_int(m.rows) or not is_int(m.cols) or m.rows < 0 or m.cols < 0 then
        error(sformat("matrix.%s: %s has invalid dimensions (rows=%s, cols=%s)", who, argname,
            tostring(m.rows), tostring(m.cols)), 3)
    end
    if type(m.data) ~= "table" then
        error(sformat("matrix.%s: %s.data must be a flat table, got %s", who, argname, type(m.data)), 3)
    end
    return m
end

local function ensure_len(m, who, argname)
    local total = m.rows * m.cols
    if total > 0 and m.data[total] == nil then
        error(sformat("matrix.%s: %s is declared %dx%d (%d elements) but data[%d] is nil",
            who, argname, m.rows, m.cols, total, total), 3)
    end
    return total
end

local function check_same_shape(a, b, who)
    if a.rows ~= b.rows or a.cols ~= b.cols then
        error(sformat("matrix.%s: size mismatch, a is %dx%d but b is %dx%d",
            who, a.rows, a.cols, b.rows, b.cols), 3)
    end
end

local function alloc(rows, cols)
    return { rows = rows, cols = cols, data = {} }
end

local function target(out, rows, cols, who)
    if out == nil then return alloc(rows, cols) end
    check_matrix(out, who, "out")
    if out.rows ~= rows or out.cols ~= cols then
        error(sformat("matrix.%s: out is %dx%d but result is %dx%d",
            who, out.rows, out.cols, rows, cols), 3)
    end
    return out
end

function matrix.backends()
    local list = {}
    for i = 1, #backends do
        local e = backends[i]
        list[i] = { name = e.name, available = e.available, reason = e.reason,
                    errors = e.errors, last_error = e.last_error }
    end
    return list
end

function matrix.use(name)
    if name == nil or name == "auto" then
        matrix.accel.force = nil
        matrix.accel.enabled = true
        return "auto"
    end
    if name == "lua" or name == false then
        matrix.accel.force = "lua"
        return "lua"
    end
    for i = 1, #backends do
        if backends[i].name == name then
            matrix.accel.force = name
            matrix.accel.enabled = true
            return name
        end
    end
    error(sformat("matrix.use: unknown backend '%s' (expected 'auto', 'lua', %s)",
        tostring(name), concat(BACKEND_NAMES, ", ")), 2)
end

local function backend_failed(entry, msg)
    entry.errors = entry.errors + 1
    entry.last_error = msg
    if entry.errors >= (matrix.accel.max_errors or 3) then
        entry.available = false
        entry.reason = "disabled after " .. entry.errors .. " failures: " .. msg
    end
end

local function try_backend(entry, ad, m, k, bd, n)
    local ok, res = pcall(entry.module.matmul, ad, m, k, bd, n)
    if not ok then
        local msg = tostring(res)
        backend_failed(entry, msg:match("^[^\n]*") or msg)
        return nil
    end
    if type(res) ~= "table" then
        backend_failed(entry, "matmul returned " .. type(res) .. ", expected flat table")
        return nil
    end
    local last = m * n
    if res[last] == nil or type(res[1]) ~= "number" then
        backend_failed(entry, sformat("matmul returned a malformed result (expected %d numbers)", last))
        return nil
    end
    return res
end

function matrix.new(rows, cols)
    if not is_int(rows) or not is_int(cols) or rows < 0 or cols < 0 then
        error(sformat("matrix.new: rows and cols must be non-negative integers (got %s, %s)",
            tostring(rows), tostring(cols)), 2)
    end
    local m = { rows = rows, cols = cols, data = {} }
    local d = m.data
    for i = 1, rows * cols do
        d[i] = 0
    end
    return m
end

function matrix.zeros(rows, cols)
    return matrix.new(rows, cols)
end

function matrix.fill(rows, cols, value)
    if type(value) ~= "number" then
        error("matrix.fill: value must be a number, got " .. type(value), 2)
    end
    local m = matrix.new(rows, cols)
    local d = m.data
    for i = 1, rows * cols do
        d[i] = value
    end
    return m
end

function matrix.identity(n)
    local m = matrix.new(n, n)
    for i = 1, n do
        m.data[(i - 1) * n + i] = 1
    end
    return m
end

function matrix.from(t, rows, cols)
    if type(t) ~= "table" then
        error("matrix.from: expected a table, got " .. type(t), 2)
    end
    if rows ~= nil or cols ~= nil then
        if not is_int(rows) or not is_int(cols) or rows < 0 or cols < 0 then
            error(sformat("matrix.from: rows/cols must be non-negative integers (got %s, %s)",
                tostring(rows), tostring(cols)), 2)
        end
        local total = rows * cols
        if total > 0 and t[total] == nil then
            error(sformat("matrix.from: flat table has fewer than %d elements for a %dx%d matrix",
                total, rows, cols), 2)
        end
        local m = alloc(rows, cols)
        local d = m.data
        for i = 1, total do
            local v = t[i]
            if type(v) ~= "number" then
                error(sformat("matrix.from: element %d is %s, expected number", i, type(v)), 2)
            end
            d[i] = v
        end
        return m
    end
    local nrows = #t
    if nrows == 0 then
        error("matrix.from: cannot build a matrix from an empty table", 2)
    end
    local first = t[1]
    if type(first) ~= "table" then
        error("matrix.from: row 1 must be a table, got " .. type(first) ..
              " (for a flat table use matrix.from(t, rows, cols))", 2)
    end
    local ncols = #first
    if ncols == 0 then
        error("matrix.from: row 1 is empty", 2)
    end
    local m = alloc(nrows, ncols)
    local d = m.data
    for i = 1, nrows do
        local row = t[i]
        if type(row) ~= "table" then
            error(sformat("matrix.from: row %d must be a table, got %s", i, type(row)), 2)
        end
        if #row ~= ncols then
            error(sformat("matrix.from: ragged input, row %d has %d columns but row 1 has %d",
                i, #row, ncols), 2)
        end
        local base = (i - 1) * ncols
        for j = 1, ncols do
            local v = row[j]
            if type(v) ~= "number" then
                error(sformat("matrix.from: element (%d,%d) is %s, expected number", i, j, type(v)), 2)
            end
            d[base + j] = v
        end
    end
    return m
end

function matrix.reshape(a, rows, cols)
    check_matrix(a, "reshape", "a")
    if not is_int(rows) or not is_int(cols) or rows < 0 or cols < 0 then
        error("matrix.reshape: rows and cols must be non-negative integers", 2)
    end
    local total = ensure_len(a, "reshape", "a")
    if rows * cols ~= total then
        error(sformat("matrix.reshape: cannot reshape %dx%d (%d elements) into %dx%d (%d elements)",
            a.rows, a.cols, total, rows, cols, rows * cols), 2)
    end
    local m = alloc(rows, cols)
    local ad, md = a.data, m.data
    for i = 1, total do
        md[i] = ad[i]
    end
    return m
end

function matrix.get(m, i, j)
    if type(m) ~= "table" or type(m.data) ~= "table" then
        error("matrix.get: first argument is not a matrix", 2)
    end
    if i < 1 or i > m.rows or j < 1 or j > m.cols then
        error(sformat("matrix.get: index (%s,%s) out of bounds for %dx%d matrix",
            tostring(i), tostring(j), m.rows, m.cols), 2)
    end
    return m.data[(i - 1) * m.cols + j]
end

function matrix.set(m, i, j, val)
    if type(m) ~= "table" or type(m.data) ~= "table" then
        error("matrix.set: first argument is not a matrix", 2)
    end
    if i < 1 or i > m.rows or j < 1 or j > m.cols then
        error(sformat("matrix.set: index (%s,%s) out of bounds for %dx%d matrix",
            tostring(i), tostring(j), m.rows, m.cols), 2)
    end
    if type(val) ~= "number" then
        error("matrix.set: value must be a number, got " .. type(val), 2)
    end
    m.data[(i - 1) * m.cols + j] = val
    return m
end

function matrix.tostring(m, fmt)
    check_matrix(m, "tostring", "m")
    fmt = fmt or "%8.3f"
    local lines, cells = {}, {}
    local d, cols = m.data, m.cols
    for i = 1, m.rows do
        local base = (i - 1) * cols
        for j = 1, cols do
            local v = d[base + j]
            if type(v) == "number" then
                cells[j] = sformat(fmt, v)
            else
                cells[j] = sformat("%8s", tostring(v))
            end
        end
        for j = cols + 1, #cells do
            cells[j] = nil
        end
        lines[i] = concat(cells)
    end
    return concat(lines, "\n")
end

function matrix.print(m, fmt)
    local s = matrix.tostring(m, fmt)
    if s == "" then
        print(sformat("<empty %dx%d matrix>", m.rows, m.cols))
    else
        print(s)
    end
    return m
end

function matrix.add(a, b, out)
    check_matrix(a, "add", "a")
    check_matrix(b, "add", "b")
    check_same_shape(a, b, "add")
    local total = ensure_len(a, "add", "a")
    ensure_len(b, "add", "b")
    local result = target(out, a.rows, a.cols, "add")
    local ad, bd, rd = a.data, b.data, result.data
    for k = 1, total do
        rd[k] = ad[k] + bd[k]
    end
    return result
end

function matrix.sub(a, b, out)
    check_matrix(a, "sub", "a")
    check_matrix(b, "sub", "b")
    check_same_shape(a, b, "sub")
    local total = ensure_len(a, "sub", "a")
    ensure_len(b, "sub", "b")
    local result = target(out, a.rows, a.cols, "sub")
    local ad, bd, rd = a.data, b.data, result.data
    for k = 1, total do
        rd[k] = ad[k] - bd[k]
    end
    return result
end

function matrix.hadamard(a, b, out)
    check_matrix(a, "hadamard", "a")
    check_matrix(b, "hadamard", "b")
    check_same_shape(a, b, "hadamard")
    local total = ensure_len(a, "hadamard", "a")
    ensure_len(b, "hadamard", "b")
    local result = target(out, a.rows, a.cols, "hadamard")
    local ad, bd, rd = a.data, b.data, result.data
    for k = 1, total do
        rd[k] = ad[k] * bd[k]
    end
    return result
end

function matrix.multiply(a, b)
    check_matrix(a, "multiply", "a")
    check_matrix(b, "multiply", "b")
    if a.cols ~= b.rows then
        error(sformat("matrix.multiply: inner dimension mismatch, a is %dx%d and b is %dx%d (a.cols %d must equal b.rows %d)",
            a.rows, a.cols, b.rows, b.cols, a.cols, b.rows), 2)
    end

    local m, k, n = a.rows, a.cols, b.cols
    if m == 0 or n == 0 or k == 0 then
        matrix.last_backend = "lua"
        return matrix.new(m, n)
    end
    ensure_len(a, "multiply", "a")
    ensure_len(b, "multiply", "b")

    local ad, bd = a.data, b.data
    local force = matrix.accel.force

    if matrix.accel.enabled and force ~= "lua" then
        local work = m * k * n
        local threshold = matrix.accel.threshold or 0
        if force ~= nil or work >= threshold then
            for i = 1, #backends do
                local entry = backends[i]
                if entry.available and (force == nil or force == entry.name) then
                    local res = try_backend(entry, ad, m, k, bd, n)
                    if res ~= nil then
                        matrix.last_backend = entry.name
                        return { rows = m, cols = n, data = res }
                    end
                end
            end
            if force ~= nil and force ~= "lua" then
                error(sformat("matrix.multiply: backend '%s' was forced via matrix.use() but is unavailable", force), 2)
            end
        end
    end

    matrix.last_backend = "lua"
    local result = alloc(m, n)
    local rd = result.data

    for i = 1, m do
        local a_base = (i - 1) * k
        local r_base = (i - 1) * n
        local av = ad[a_base + 1]
        local b_base = 0
        if av ~= 0 then
            for j = 1, n do
                rd[r_base + j] = av * bd[j]
            end
        else
            for j = 1, n do
                rd[r_base + j] = 0
            end
        end
        for p = 2, k do
            av = ad[a_base + p]
            if av ~= 0 then
                b_base = (p - 1) * n
                for j = 1, n do
                    local idx = r_base + j
                    rd[idx] = rd[idx] + av * bd[b_base + j]
                end
            end
        end
    end
    return result
end

function matrix.scale(a, n, out)
    check_matrix(a, "scale", "a")
    if type(n) ~= "number" then
        error("matrix.scale: scalar must be a number, got " .. type(n), 2)
    end
    local total = ensure_len(a, "scale", "a")
    local result = target(out, a.rows, a.cols, "scale")
    local ad, rd = a.data, result.data
    for k = 1, total do
        rd[k] = ad[k] * n
    end
    return result
end

function matrix.map(a, fn, out)
    check_matrix(a, "map", "a")
    if type(fn) ~= "function" then
        error("matrix.map: fn must be a function, got " .. type(fn), 2)
    end
    local total = ensure_len(a, "map", "a")
    local result = target(out, a.rows, a.cols, "map")
    local ad, rd = a.data, result.data
    for k = 1, total do
        rd[k] = fn(ad[k])
    end
    return result
end

function matrix.transpose(a, out)
    check_matrix(a, "transpose", "a")
    ensure_len(a, "transpose", "a")
    local rows, cols = a.rows, a.cols
    local result = target(out, cols, rows, "transpose")
    local ad, rd = a.data, result.data
    for i = 1, rows do
        local a_base = (i - 1) * cols
        local idx = i
        for j = 1, cols do
            rd[idx] = ad[a_base + j]
            idx = idx + rows
        end
    end
    return result
end

function matrix.copy(a, out)
    check_matrix(a, "copy", "a")
    local total = ensure_len(a, "copy", "a")
    local result = target(out, a.rows, a.cols, "copy")
    local ad, rd = a.data, result.data
    for k = 1, total do
        rd[k] = ad[k]
    end
    return result
end

function matrix.equals(a, b, tol)
    check_matrix(a, "equals", "a")
    check_matrix(b, "equals", "b")
    if a.rows ~= b.rows or a.cols ~= b.cols then return false end
    tol = tol or 0
    local total = a.rows * a.cols
    local ad, bd = a.data, b.data
    for k = 1, total do
        local x, y = ad[k], bd[k]
        if type(x) ~= "number" or type(y) ~= "number" then return false end
        local diff = x - y
        if diff < 0 then diff = -diff end
        if diff > tol then return false end
    end
    return true
end

return matrix
