local matrix = require("matrix")
local Tensor = require("tensor2")

local passes, fails = 0, 0

local function assert_equal(a, b, name, tol)
    tol = tol or 1e-5
    if type(a) == "table" and type(b) == "table" then
        if #a ~= #b then fails = fails + 1; print("FAIL: " .. name .. " (length mismatch)"); return end
        for i = 1, #a do
            if math.abs(a[i] - b[i]) > tol then
                fails = fails + 1
                print("FAIL: " .. name .. " (mismatch at index " .. i .. ": " .. a[i] .. " ~= " .. b[i] .. ")")
                return
            end
        end
        passes = passes + 1
        print("PASS: " .. name)
    elseif math.abs(a - b) > tol then
        fails = fails + 1
        print("FAIL: " .. name .. " (" .. tostring(a) .. " ~= " .. tostring(b) .. ")")
    else
        passes = passes + 1
        print("PASS: " .. name)
    end
end

print("=========================================")
print("  LanternL Test Suite")
print("=========================================")

-- 1. Matrix Operations
local A = matrix.from({1, 2, 3, 4}, 2, 2)
local B = matrix.from({5, 6, 7, 8}, 2, 2)
local C = matrix.multiply(A, B)
assert_equal(C.data, {19, 22, 43, 50}, "Matrix Multiply (2x2)")

local T = matrix.transpose(A)
assert_equal(T.data, {1, 3, 2, 4}, "Matrix Transpose")

-- 2. Tensor Autograd
local x = Tensor.from({1.0, 2.0, 3.0})
local w = Tensor.from({0.5, 0.5, 0.5})
local y = Tensor.matmul(x, w) -- y = x * w^T = (1*0.5 + 2*0.5 + 3*0.5) = 3.0
y:backward()
assert_equal(y.data.data[1], 3.0, "Tensor Forward (Dot Product)")
assert_equal(w.grad.data[1], 1.0, "Tensor Backward (Grad Accumulation)")

-- 3. Softmax Stability
local logits = Tensor.from({1000.0, 1000.0, 1000.0})
local probs = Tensor.softmax_rows(logits)
-- Since all are equal, probability should be 1/3
assert_equal(probs.data.data[1], 0.3333333, "Softmax Numerical Stability", 1e-4)

-- 4. Cross Entropy
local ce_logits = Tensor.from({2.0, 1.0, 0.1})
local loss = Tensor.cross_entropy(ce_logits, {1}) -- Target is index 1
-- loss = -log(softmax(2.0))
assert_equal(loss.data.data[1], 0.417, "Cross Entropy Loss", 1e-2)

print("=========================================")
print(string.format("Results: %d passed, %d failed", passes, fails))
if fails > 0 then os.exit(1) else os.exit(0) end