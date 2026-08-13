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

-- Add these to the bottom of tests.lua, right before the final print results

print("\n--- Advanced Tests ---")

-- 5. Tokenizer Round-Trip
local Tokenizer = require("tokenizer")
local tok = Tokenizer.new { text = "hello world", vocab_size = 100 }
tok:train()
local ids = tok:encode("hello")
assert_equal(#ids > 0, true, "Tokenizer Encode Produces IDs")
local txt = tok:decode(ids)
assert_equal(type(txt), "string", "Tokenizer Decode Returns String")

-- 6. Bad Config Rejection
local LMTrain = require("lmtrain")
local ok, err = pcall(function()
    -- 64 % 3 != 0, should throw an error
    LMTrain.new { dim = 64, heads = 3, data = "test data" }
end)
assert_equal(ok, false, "Bad Config Rejection (dim % heads)")

local ok2 = pcall(function()
    -- max_seq < 2, should throw an error
    LMTrain.new { data = "test", max_seq = 1 }
end)
assert_equal(ok2, false, "Bad Config Rejection (max_seq < 2)")

-- 7. RoPE Angle Correctness
-- For position 0, cos(0) = 1, sin(0) = 0, so RoPE should be identity
local RoPE = require("rope")
local rope = RoPE.new { head_dim = 4, base = 10000 }
local cos_0, sin_0 = rope:rows(1)
assert_equal(cos_0[1][1], 1.0, "RoPE Angle pos 0 (cos=1)")
assert_equal(sin_0[1][1], 0.0, "RoPE Angle pos 0 (sin=0)")

-- 8. Gradient NaN Check
local nan_tensor = Tensor.from({math.huge, math.huge})
local w = Tensor.from({0.5, 0.5})
local y = Tensor.matmul(nan_tensor, w)
-- We expect a number, but if it's NaN, we catch it
local is_nan = (y.data.data[1] ~= y.data.data[1])
assert_equal(is_nan, true, "NaN Detection in Forward Pass")

print("=========================================")
print(string.format("Results: %d passed, %d failed", passes, fails))
if fails > 0 then os.exit(1) else os.exit(0) end
