--  lmlog.lua -- wall clock, TTY detection, stdout buffering policy and the
--  shared progress reporter used by BOTH backends.
--
--  Output policy lives here (not in lmtrain.lua) because this is the only
--  module every output path goes through.  It used to be duplicated twice
--  in lmtrain.lua, which meant a consumer that required lmlog directly got
--  whatever buffering libc happened to pick.

local ffi = require("ffi")

local L = {}

local floor, exp, huge = math.floor, math.exp, math.huge
local sformat, srep = string.format, string.rep

-- ---------------------------------------------------------------- clock ----
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

-- ------------------------------------------------------------ tty probe ----
-- Was written as `(A) and (pcall(f) and f()) or false`, which called
-- isatty() twice and leaned on `or false` to normalise.  Same result,
-- stated once.
do
    pcall(ffi.cdef, "int isatty(int fd);")
    local tty = false
    if ffi.os ~= "Windows" then
        local ok, r = pcall(function() return ffi.C.isatty(1) end)
        tty = ok and (r == 1)
    end
    local term = os.getenv("TERM")
    if term == "dumb" then tty = false end
    L.is_tty = tty
end

-- ------------------------------------------------- stdout buffering -------
--  PROBLEM 4, library side.
--
--  On a TTY we want unbuffered so the in-place "\r" bar (which never emits
--  a newline) appears as it is written.  On a pipe -- which is exactly what
--  `%%script luajit` and `!luajit` both hand us -- line buffering is the
--  right mode: every complete line is flushed atomically at the "\n", with
--  one write(2) per line instead of one per byte.  The reporter also calls
--  flush() explicitly after every emit, so either mode streams; "line" just
--  costs far less on a hot loop.
--
--  Idempotent: safe to call from any entry point, and called on require.
local _vbuf_done = false
function L.setup_stdout(mode)
    if _vbuf_done and mode == nil then return end
    _vbuf_done = true
    mode = mode or (L.is_tty and "no" or "line")
    pcall(function() io.stdout:setvbuf(mode) end)
    return mode
end
L.setup_stdout()

--- Default emit interval.  A piped run has no in-place overwrite, so every
--- update is a permanent line; 0.2s over a 4h run is ~72k lines of notebook
--- output.  Slower default off-TTY, still live.
function L.default_interval()
    return L.is_tty and 0.2 or 1.0
end

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

function L.fmt_bytes(n)
    if n >= 1073741824 then return sformat("%.2f GB", n / 1073741824) end
    if n >= 1048576    then return sformat("%.1f MB", n / 1048576) end
    if n >= 1024       then return sformat("%.1f KB", n / 1024) end
    return sformat("%d B", n)
end

-- --------------------------------------------------------- reporter -------
local R = {}
R.__index = R

--- The reporter that currently owns the cursor, if any.  L.say() routes
--- through it so a library message can never land in the middle of an
--- in-place bar.
L.active = nil

--- opts: total_steps, bar_style, bar_len, interval (seconds),
---       inplace (bool), force_inplace (bool), stream (io handle)
function L.new(opts)
    opts = opts or {}
    local self = setmetatable({}, R)
    self.total_steps = opts.total_steps or 0
    self.bar_style   = opts.bar_style or 1
    self.bar_len     = opts.bar_len or 20
    self.interval    = opts.interval or L.default_interval()
    -- force_inplace lets a caller opt into "\r" rendering on a pipe (some
    -- terminals-in-a-notebook handle it); default stays TTY-gated.
    if opts.force_inplace then
        self.inplace = true
    else
        self.inplace = (opts.inplace ~= false) and L.is_tty
    end
    self.out         = opts.stream or io.stdout
    self.t0          = now()
    self.last_emit   = -1e9
    self.win_t       = self.t0
    self.win_tok     = 0
    self.rate        = 0
    self.width       = 0
    self.dirty       = false
    L.active = self
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
    if L.active == self then L.active = nil end
end

--- Module-level message.  Routes through the live reporter when there is
--- one so it cannot land mid-bar; otherwise a plain flushed write.  Both
--- backends now use this instead of bare print()/io.write().
function L.say(text)
    local a = L.active
    if a then return a:say(text) end
    io.stdout:write(text, "\n")
    io.stdout:flush()
end

return L
