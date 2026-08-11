local matrix = {}

function matrix.new(rows, cols)
    local m = {rows = rows, cols = cols, data = {}}
    local total = rows * cols
    for i = 1, total do
        m.data[i] = 0
    end
    return m
end

function matrix.from(t)
    local rows = #t
    local cols = #t[1]
    local m = matrix.new(rows, cols)
    for i = 1, rows do
        local row = t[i]
        local base = (i - 1) * cols
        for j = 1, cols do
            m.data[base + j] = row[j]
        end
    end
    return m
end

function matrix.get(m, i, j)
    return m.data[(i - 1) * m.cols + j]
end

function matrix.set(m, i, j, val)
    m.data[(i - 1) * m.cols + j] = val
end

function matrix.print(m)
    for i = 1, m.rows do
        local base = (i - 1) * m.cols
        local row = ""
        for j = 1, m.cols do
            row = row .. string.format("%8.3f", m.data[base + j])
        end
        print(row)
    end
end

function matrix.add(a, b)
    assert(a.rows == b.rows and a.cols == b.cols, "add: size mismatch")
    local result = matrix.new(a.rows, a.cols)
    local total = a.rows * a.cols
    for k = 1, total do
        result.data[k] = a.data[k] + b.data[k]
    end
    return result
end

function matrix.sub(a, b)
    assert(a.rows == b.rows and a.cols == b.cols, "sub: size mismatch")
    local result = matrix.new(a.rows, a.cols)
    local total = a.rows * a.cols
    for k = 1, total do
        result.data[k] = a.data[k] - b.data[k]
    end
    return result
end

function matrix.multiply(a, b)
    assert(a.cols == b.rows, "multiply: a.cols must equal b.rows")
    local result = matrix.new(a.rows, b.cols)
    local ad, bd, rd = a.data, b.data, result.data
    local acols, bcols = a.cols, b.cols

    for i = 1, a.rows do
        local a_base = (i - 1) * acols
        local r_base = (i - 1) * bcols
        for j = 1, bcols do
            local sum = 0
            for k = 1, acols do
                sum = sum + ad[a_base + k] * bd[(k - 1) * bcols + j]
            end
            rd[r_base + j] = sum
        end
    end
    return result
end

function matrix.scale(a, n)
    local result = matrix.new(a.rows, a.cols)
    local total = a.rows * a.cols
    for k = 1, total do
        result.data[k] = a.data[k] * n
    end
    return result
end

function matrix.transpose(a)
    local result = matrix.new(a.cols, a.rows)
    for i = 1, a.rows do
        local a_base = (i - 1) * a.cols
        for j = 1, a.cols do
            result.data[(j - 1) * a.rows + i] = a.data[a_base + j]
        end
    end
    return result
end

function matrix.copy(a)
    local result = matrix.new(a.rows, a.cols)
    local total = a.rows * a.cols
    for k = 1, total do
        result.data[k] = a.data[k]
    end
    return result
end

return matrix