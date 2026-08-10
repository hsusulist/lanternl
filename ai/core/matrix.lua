local matrix = {}

function matrix.new(rows, cols)
    local m = {rows = rows, cols = cols, data = {}}
    for i = 1, rows do
        m.data[i] = {}
        for j = 1, cols do
            m.data[i][j] = 0
        end
    end
    return m
end

function matrix.from(t)
    local m = {rows = #t, cols = #t[1], data = t}
    return m
end

function matrix.print(m)
    for i = 1, m.rows do
        local row = ""
        for j = 1, m.cols do
            row = row .. string.format("%8.3f", m.data[i][j])
        end
        print(row)
    end
end

function matrix.add(a, b)
    assert(a.rows == b.rows and a.cols == b.cols, "add: size mismatch")
    local result = matrix.new(a.rows, a.cols)
    for i = 1, a.rows do
        for j = 1, a.cols do
            result.data[i][j] = a.data[i][j] + b.data[i][j]
        end
    end
    return result
end

function matrix.sub(a, b)
    assert(a.rows == b.rows and a.cols == b.cols, "sub: size mismatch")
    local result = matrix.new(a.rows, a.cols)
    for i = 1, a.rows do
        for j = 1, a.cols do
            result.data[i][j] = a.data[i][j] - b.data[i][j]
        end
    end
    return result
end

function matrix.multiply(a, b)
    assert(a.cols == b.rows, "multiply: a.cols must equal b.rows")
    local result = matrix.new(a.rows, b.cols)
    for i = 1, a.rows do
        for j = 1, b.cols do
            local sum = 0
            for k = 1, a.cols do
                sum = sum + a.data[i][k] * b.data[k][j]
            end
            result.data[i][j] = sum
        end
    end
    return result
end

-- Scale matrix by a number
function matrix.scale(a, n)
    local result = matrix.new(a.rows, a.cols)
    for i = 1, a.rows do
        for j = 1, a.cols do
            result.data[i][j] = a.data[i][j] * n
        end
    end
    return result
end

function matrix.transpose(a)
    local result = matrix.new(a.cols, a.rows)
    for i = 1, a.rows do
        for j = 1, a.cols do
            result.data[j][i] = a.data[i][j]
        end
    end
    return result
end

function matrix.copy(a)
    local result = matrix.new(a.rows, a.cols)
    for i = 1, a.rows do
        for j = 1, a.cols do
            result.data[i][j] = a.data[i][j]
        end
    end
    return result
end

return matrix