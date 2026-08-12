
local ok_luatl, luaTL = pcall(require, "luaTL")
if not ok_luatl then
    return { available = false, reason = "require('luaTL') failed: " .. tostring(luaTL) }
end

local ok_init, init_err = pcall(function() luaTL.init(0) end)
if not ok_init then
    return { available = false, reason = "luaTL.init(0) failed: " .. tostring(init_err) }
end

local adapter = { available = true, device = luaTL.device }

function adapter.matmul(a_flat, m, k, b_flat, n)
    local A = luaTL.Tensor(a_flat, { m, k })
    local B = luaTL.Tensor(b_flat, { k, n })
    A:to_gpu(); B:to_gpu()

    local C = A:matmul(B)
    C:to_cpu()

    local result = {}
    for i = 0, m * n - 1 do result[i + 1] = C.host[i] end

    A:free(); B:free(); C:free()
    return result
end

function adapter.softmax(x_flat, rows, cols)
    local X = luaTL.Tensor(x_flat, { rows, cols })
    X:to_gpu()

    local Out = X:softmax()
    Out:to_cpu()

    local result = {}
    for i = 0, rows * cols - 1 do result[i + 1] = Out.host[i] end

    X:free(); Out:free()
    return result
end

-- Returns rmsnorm output flat array. NOTE: backward still needs the
-- per-row scale (r = rsqrt(mean_sq + eps)), which this kernel doesn't
-- expose separately, so rmsnorm.lua recomputes that small reduction on
-- CPU regardless of whether this GPU path is used.
function adapter.rmsnorm(x_flat, rows, cols, weight_flat, eps)
    local X = luaTL.Tensor(x_flat, { rows, cols })
    X:to_gpu()

    local W = nil
    if weight_flat then
        W = luaTL.Tensor(weight_flat, { 1, cols })
        W:to_gpu()
    end

    local Out = X:rmsnorm(W, eps)
    Out:to_cpu()

    local result = {}
    for i = 0, rows * cols - 1 do result[i + 1] = Out.host[i] end

    X:free(); Out:free()
    if W then W:free() end
    return result
end

return adapter