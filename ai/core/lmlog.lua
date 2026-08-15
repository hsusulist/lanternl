-- =====================================================================
--  lmlog.lua — wall-clock, time-throttled training progress reporter
--
--  Why this exists
--  ---------------
--   * The old logger fired once per EPOCH.  A 90-second epoch meant 90
--     seconds of a frozen terminal.  This one fires on a WALL-CLOCK
--     interval (default 200 ms) during the epoch, so you see the bar
--     move within the first fraction of a second.
--   * os.clock() is CPU time.  While the GPU works, or while the OS
--     deschedules you, it barely advances -- so throughput and ETA were
--     both wildly overstated.  We use gettimeofday (falling back to
--     os.time) for real elapsed seconds.
--   * Throughput is a rolling window over the last ~2 s, not a
--     since-the-beginning average, so it reacts to what is happening now.
-- =====================================================================

local ffi = require("ffi")

local L = {}

local floor, exp, huge = math.floor, math.exp, math.huge
local sformat, srep = string.format, string.rep

-- ---- wall clock ------------------------------------------------------
local now
do
    local ok = pcall(ffi.cdef, [[
        typedef long time_t2_;
        struct luatl_tv { long tv_sec; long tv_usec; };
        int gettimeofday(struct luatl_tv *tv, void *tz);
    ]])
    if ok and ffi.os ~= "Windows" then
        local tv = ffi.new("struct luatl_tv")
        now = function()
            ffi.C.gettimeofday(tv, nil)
            return tonumber(tv.tv_sec) + tonumber(tv.tv_usec) * 1e-6
        end
    else
        local ok2 = pcall(ffi.cdef, [[
            int QueryPerformanceCounter(int64_t *c);
            int QueryPerformanceFrequency(int64_t *f);
        ]])
        if ok2 then
            local c, f = ffi.new("int64_t[1]"), ffi.new("int64_t[1]")
            if ffi.C.QueryPerformanceFrequency(f) ~= 0 then
                local freq = tonumber(f[0])
                now = function()
                    ffi.C.QueryPerformanceCounter(c)
                    return tonumber(c[0]) / freq
                end
            end
        end
    end
    if not now then
        local t0 = os.time()
        now = function() return os.difftime(os.time(), t0) + os.clock() % 1 end
    end
end
L.now = now

pcall(ffi.cdef, "int isatty(int fd);")
L.is_tty = (ffi.os ~= "Windows") and (pcall(function()
    return ffi.C.isatty(1) == 1
end) and ffi.C.isatty(1) == 1) or false

local BAR = {
    [1] = { fill = "#", empty = "-" },
    [2] = { fill = "\226\150\147", empty = "\226\150\145" },
    [3] = { fill = "=", empty = " " },
}
L.BAR = BAR

function L.fmt_time(s)
    if s ~= s or s == huge then return "  --  " end
    if s < 60 then return sformat("%4.1fs", s) end
    if s < 3600 then return sformat("%dm%02ds", floor(s / 60), floor(s % 60)) end
    return sformat("%dh%02dm", floor(s / 3600), floor((s % 3600) / 60))
end

function L.fmt_count(n)
    if n >= 1e9 then return sformat("%.2fB", n / 1e9) end
    if n >= 1e6 then return sformat("%.1fM", n / 1e6) end
    if n >= 1e3 then return sformat("%.1fK", n / 1e3) end
    return sformat("%d", n)
end

-- =====================================================================
--  Reporter object
-- =====================================================================
local R = {}
R.__index = R

--- opts: total_steps, bar_style, bar_len, interval (seconds),
---       inplace (bool), stream (io handle)
function L.new(opts)
    opts = opts or {}
    local self = setmetatable({}, R)
    self.total_steps = opts.total_steps or 0
    self.bar_style   = opts.bar_style or 1
    self.bar_len     = opts.bar_len or 20
    self.interval    = opts.interval or 0.2
    self.inplace     = (opts.inplace ~= false) and L.is_tty
    self.out         = opts.stream or io.stdout
    self.t0          = now()
    self.last_emit   = -1e9
    self.win_t       = self.t0
    self.win_tok     = 0
    self.rate        = 0
    self.width       = 0
    self.dirty       = false
    return self
end

function R:tick(tokens) self.win_tok = self.win_tok + (tokens or 0) end

function R:elapsed() return now() - self.t0 end

function R:_rate()
    local t = now()
    local dt = t - self.win_t
    if dt >= 1.0 then
        local r = self.win_tok / dt
        self.rate = (self.rate > 0) and (0.6 * self.rate + 0.4 * r) or r
        self.win_t, self.win_tok = t, 0
    end
    return self.rate
end

--- st fields: step, epoch, epochs, batch, nbatch, loss, acc, lr,
---            val_loss, gnorm, best, tflops
--- force = emit regardless of the throttle (end of epoch, final step).
function R:update(st, force)
    local t = now()
    if not force and (t - self.last_emit) < self.interval then return false end
    self.last_emit = t

    local style = self.bar_style
    if type(style) ~= "table" then style = BAR[style] or BAR[1] end

    local frac = 0
    if self.total_steps > 0 then frac = (st.step or 0) / self.total_steps end
    if frac < 0 then frac = 0 elseif frac > 1 then frac = 1 end
    local filled = floor(frac * self.bar_len)
    local bar = srep(style.fill, filled) .. srep(style.empty, self.bar_len - filled)

    local el = t - self.t0
    local eta = huge
    if frac > 0.0005 then eta = el * (1 - frac) / frac end

    local loss = st.loss or 0
    local ppl = exp(loss < 30 and loss or 30)

    local parts = {}
    parts[#parts + 1] = sformat("[%s] %3.0f%%", bar, frac * 100)
    if st.epochs and st.epochs > 0 then
        parts[#parts + 1] = sformat("ep %d/%d", st.epoch or 0, st.epochs)
    end
    if st.nbatch and st.nbatch > 1 then
        parts[#parts + 1] = sformat("b %d/%d", st.batch or 0, st.nbatch)
    end
    parts[#parts + 1] = sformat("loss %.4f", loss)
    parts[#parts + 1] = sformat("ppl %7.2f", ppl)
    if st.acc then parts[#parts + 1] = sformat("acc %5.1f%%", st.acc) end
    if st.lr then parts[#parts + 1] = sformat("lr %.2e", st.lr) end
    parts[#parts + 1] = sformat("%s tok/s", L.fmt_count(self:_rate()))
    if st.tflops and st.tflops > 0 then
        parts[#parts + 1] = sformat("%.1f GF/s", st.tflops * 1000)
    end
    if st.gnorm and st.gnorm > 0 then
        parts[#parts + 1] = sformat("|g| %.2f", st.gnorm)
    end
    if st.val_loss then parts[#parts + 1] = sformat("val %.4f", st.val_loss) end
    parts[#parts + 1] = sformat("eta %s", L.fmt_time(eta))
    if st.best then parts[#parts + 1] = "*" end

    local line = table.concat(parts, " | ")

    if self.inplace then
        local pad = self.width - #line
        self.out:write("\r", line, pad > 0 and srep(" ", pad) or "")
        self.width = #line
        self.dirty = true
    else
        self.out:write(line, "\n")
    end
    self.out:flush()
    return true
end

--- Print a permanent line (epoch summary, warning) without eating the bar.
function R:say(text)
    if self.dirty then self.out:write("\n") self.dirty = false self.width = 0 end
    self.out:write(text, "\n")
    self.out:flush()
end

function R:finish()
    if self.dirty then self.out:write("\n") self.dirty = false self.width = 0 end
    self.out:flush()
end

return L
