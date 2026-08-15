-- =====================================================================
--  luatl_adapter.lua — lanternl <-> luaTL bridge
--
--  Replaces the old luatl_adapter.lua.  100% backward compatible:
--    M.matmul(a_flat, m, k, b_flat, n) -> flat 1-indexed table (m*n)
--    M.rmsnorm(x_flat, rows, cols [, w_flat, eps]) -> flat table
--    M.softmax(x_flat, rows, cols)                 -> flat table
--    M.softmax_causal(x_flat, rows, cols [, scale])-> flat table
--
--  NEW (opt-in, nothing breaks if unused):
--    * The same functions accept GPU HANDLES instead of flat tables.
--    * M.pin(flat, rows, cols)  -> upload a weight ONCE, forever.
--    * Transposed-GEMM helpers for the backward pass.
--    * A resident tensor toolkit (stride views, offset pointers, int32
--      index buffers, scalar readback) used by gpu_transformer.lua.
-- =====================================================================

local M = { name = "luatl", priority = 100, available = false }

local ok, luaTL = pcall(require, "luaTL")
if not ok then M.error = tostring(luaTL) return M end
local ok2, err2 = pcall(require, "luaTL_train")
if not ok2 then M.error = tostring(err2) return M end

local ffi = require("ffi")

local okinit, initerr = pcall(function() luaTL.init(0) end)
if not okinit then M.error = tostring(initerr) return M end

M.available = true
M.device    = luaTL.device
M.luaTL     = luaTL
M.C         = luaTL.C

local F32     = luaTL.F32
local f32_ct  = ffi.typeof("float[?]")
local i32_ct  = ffi.typeof("int32_t[?]")

-- ---------------------------------------------------------------------
--  Cached host staging buffers (stops per-call cdata churn)
-- ---------------------------------------------------------------------
local stage, stage_n = {}, {}
local function host_buf(slot, n)
    if not stage[slot] or stage_n[slot] < n then
        stage[slot]   = f32_ct(n)
        stage_n[slot] = n
    end
    return stage[slot]
end

-- ---------------------------------------------------------------------
--  Cached device scratch, keyed by "rows x cols" (pool-backed)
-- ---------------------------------------------------------------------
local scratch = {}
local function dev_scratch(slot, rows, cols)
    local key = slot .. ":" .. rows .. "x" .. cols
    local t = scratch[key]
    if not t then
        t = luaTL.new(rows, cols, F32)
        scratch[key] = t
    end
    return t
end

-- ---------------------------------------------------------------------
--  Handles: a persistent GPU tensor exposed to lanternl
-- ---------------------------------------------------------------------
local Handle = {}
Handle.__index = Handle
M.HandleClass = Handle

local function is_handle(x)
    return type(x) == "table" and getmetatable(x) == Handle
end
M.is_handle = is_handle

local function new_handle(t, rows, cols)
    return setmetatable({ t = t, rows = rows, cols = cols,
                          __luatl_handle = true }, Handle)
end
M.new_handle = new_handle

--- Upload a flat Lua array once and keep it resident.
function M.pin(flat, rows, cols)
    local n = rows * cols
    if #flat ~= n then
        error(("luatl.pin: %d elements != %dx%d"):format(#flat, rows, cols), 2)
    end
    local buf = host_buf("pin", n)
    for i = 1, n do buf[i - 1] = flat[i] end
    local t = luaTL.new(rows, cols, F32)
    t:upload(buf, n)
    return new_handle(t, rows, cols)
end

--- Allocate a resident zero tensor (activations, grads, optimizer state).
function M.alloc(rows, cols)
    return new_handle(luaTL.zeros(rows, cols, F32), rows, cols)
end

function Handle:toflat()
    local n   = self.rows * self.cols
    local buf = host_buf("dl", n)
    self.t:download(buf, n)
    local out = {}
    for i = 1, n do out[i] = buf[i - 1] end
    return out
end

function Handle:upload(flat)
    local n = self.rows * self.cols
    local buf = host_buf("ul", n)
    for i = 1, n do buf[i - 1] = flat[i] end
    self.t:upload(buf, n)
    return self
end

function Handle:zero() self.t:zero() return self end
function Handle:free() if self.t then self.t:free() self.t = nil end end
function Handle:tensor() return self.t end
function Handle:shape() return self.rows, self.cols end

-- ---------------------------------------------------------------------
--  Upload helper: accept a handle OR a flat table
-- ---------------------------------------------------------------------
local function as_tensor(x, rows, cols, slot)
    if is_handle(x) then return x.t, true end
    local n = rows * cols
    if #x < n then
        error(("luatl adapter: expected %d elements, got %d"):format(n, #x), 3)
    end
    local buf = host_buf(slot, n)
    for i = 1, n do buf[i - 1] = x[i] end
    local t = dev_scratch(slot, rows, cols)
    t:upload(buf, n)
    return t, false
end

local function emit(t, rows, cols, resident)
    if resident then return new_handle(t, rows, cols) end
    local n   = rows * cols
    local buf = host_buf("out", n)
    t:download(buf, n)
    local out = {}
    for i = 1, n do out[i] = buf[i - 1] end
    return out
end

-- =====================================================================
--  LEGACY-COMPATIBLE ENTRY POINTS  (signatures unchanged)
-- =====================================================================

function M.matmul(a, m, k, b, n)
    local A, ah = as_tensor(a, m, k, "A")
    local B, bh = as_tensor(b, k, n, "B")
    local resident = ah and bh
    local Cc = resident and luaTL.new(m, n, F32) or dev_scratch("C", m, n)
    luaTL.gemm_ex(A, B, Cc)
    return emit(Cc, m, n, resident)
end

function M.rmsnorm(x, rows, cols, w, eps)
    local X, xh = as_tensor(x, rows, cols, "A")
    local W = nil
    if w then W = (as_tensor(w, 1, cols, "B")) end
    local O = xh and luaTL.new(rows, cols, F32) or dev_scratch("C", rows, cols)
    luaTL.check(luaTL.C.luaTL_rmsnorm(X.data, W and W.data or nil, O.data,
        rows, cols, eps or 1e-5, F32), "adapter.rmsnorm")
    return emit(O, rows, cols, xh)
end

function M.softmax(x, rows, cols)
    local X, xh = as_tensor(x, rows, cols, "A")
    local O = xh and luaTL.new(rows, cols, F32) or dev_scratch("C", rows, cols)
    X:softmax_ex(O, {})
    return emit(O, rows, cols, xh)
end

function M.softmax_causal(x, rows, cols, scale, group)
    local X, xh = as_tensor(x, rows, cols, "A")
    local O = xh and luaTL.new(rows, cols, F32) or dev_scratch("C", rows, cols)
    X:softmax_ex(O, { causal = true, scale = scale or 1.0, group = group or 0 })
    return emit(O, rows, cols, xh)
end

-- =====================================================================
--  Resident-mode primitives for a migrated matrix.lua / tensor2.lua
-- =====================================================================

function M.gemm(hA, hB, hC, opts) luaTL.gemm_ex(hA.t, hB.t, hC.t, opts) return hC end
function M.gemm_dx(hdY, hW, hdX, beta) luaTL.matmul_dx(hdY.t, hW.t, hdX.t, beta) return hdX end
function M.gemm_dw(hX, hdY, hdW, beta) luaTL.matmul_dw(hX.t, hdY.t, hdW.t, beta) return hdW end

function M.add_inplace(hDst, hSrc, alpha) hDst.t:add_inplace(hSrc.t, alpha) return hDst end
function M.silu_inplace(h) h.t:silu_inplace() return h end
function M.gelu_inplace(h) h.t:gelu_inplace() return h end
function M.scale_inplace(h, a) h.t:scale_inplace(a, 0) return h end

function M.rope(h, o) h.t:rope(o) return h end
function M.swiglu(ha, hb, ho, o) luaTL.swiglu(ha.t, hb.t, ho.t, o) return ho end

function M.rmsnorm_bwd(hx, hg, hdy, hdx, hdg, eps)
    hx.t:rmsnorm_bwd(hg and hg.t, hdy.t, hdx.t, hdg and hdg.t, eps)
    return hdx
end

function M.softmax_bwd(hy, hdy, hdx, o) hy.t:softmax_bwd(hdy.t, hdx.t, o) return hdx end

function M.program(cap, timing) return luaTL.Program(cap, timing) end
function M.sync() luaTL.sync() end
function M.stats() return luaTL.pool_stats() end
function M.hardware() return luaTL.hardware() end
function M.shutdown() luaTL.finalize() end

-- =====================================================================
--  RESIDENT TENSOR TOOLKIT  (new; used by gpu_transformer.lua)
--
--  Nothing above this line changed.  These are thin, allocation-free
--  helpers for code that keeps EVERYTHING in VRAM for the lifetime of a
--  model rather than for the duration of one call.
-- =====================================================================

--- Element-offset pointer into a tensor's f32 storage.
function M.offset(t, elems)
    return ffi.cast("char*", t.data) + elems * 4
end

--- Explicit stride view for gemm_ex / Program:gemm.
--- rs = row stride, cs = col stride, bs = batch stride (all in elements).
function M.view(ptr, rows, cols, rs, cs, bs)
    return { data = ptr, rows = rows, cols = cols,
             rs = rs, cs = cs, bs = bs or 0, dtype = F32 }
end

--- Column-slice view of a handle/tensor, keeping the parent leading dim.
function M.cols(t, col0, w)
    if is_handle(t) then t = t.t end
    return M.view(M.offset(t, col0), t.rows, w, t.cols, 1, 0)
end

--- Non-owning tensor over a raw pointer (so Program ops can take it).
function M.wrap(ptr, rows, cols) return luaTL.wrap(ptr, rows, cols, F32) end

--- int32 index buffer backed by f32-sized storage (4 bytes either way).
function M.ids(n)
    local h = M.alloc(n, 1)
    h.is_ids = true
    return h
end

function M.upload_i32(t, cbuf, n)
    if is_handle(t) then t = t.t end
    luaTL.check(luaTL.C.luaTL_upload_i32(t.data, cbuf, n), "adapter.upload_i32")
end

function M.download_i32(t, cbuf, n)
    if is_handle(t) then t = t.t end
    luaTL.check(luaTL.C.luaTL_download_i32(t.data, cbuf, n), "adapter.download_i32")
end

function M.download_f32(ptr, cbuf, n)
    luaTL.check(luaTL.C.luaTL_download_f32(ptr, cbuf, n), "adapter.download_f32")
end

--- Read one float out of a device tensor (used for the loss / grad norm).
local one = ffi.new("float[1]")
function M.scalar(t)
    if is_handle(t) then t = t.t end
    luaTL.check(luaTL.C.luaTL_download_f32(t.data, one, 1), "adapter.scalar")
    return one[0]
end

function M.mem()
    local f, t = luaTL.mem_info()
    return f, t
end

return M
