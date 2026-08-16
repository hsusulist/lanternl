--  lmio.lua -- checkpoint persistence + crash-safe auto-save.
--
--  Installed onto LMTrain by lmtrain.lua:  lmio(LMTrain).
--
--  ============================ FORMATS ==================================
--  v3 bin  (default when ffi is available)
--      lanternl-checkpoint 3\n
--      format=bin dtype=f32 endian=le vocab=.. dim=.. ... nparams=..\n
--      per parameter:  "<rows> <cols>\n"  then rows*cols raw LE floats
--  v3 text / v2 text
--      ... format=text ...
--      per parameter:  "<rows> <cols>|v,v,v,...\n"
--  v1 legacy: no tag line, no "r c|" prefix.  Still readable.
--
--  ======================== WHY BINARY IS THE DEFAULT ====================
--  The old text writer built one Lua string per element and table.concat'd
--  the whole tensor.  For a 134M-parameter model that is >130M transient
--  interned strings and a ~3.2 GB file at %.17g, taking minutes per write.
--  Auto-saving every epoch with that writer would have OOM'd or dominated
--  the run.  The binary path streams through a fixed 1M-element FFI buffer:
--  constant memory, ~4 bytes/param, one write(2) per chunk.
--
--  ======================== CRASH SAFETY =================================
--  Every write goes to "<path>.tmpNNNN" IN THE SAME DIRECTORY (a temp dir
--  would make rename(2) cross-device and fail with EXDEV -- Colab users
--  save to /content/drive constantly), is fully written and closed with
--  every write() and the close() result checked, and only then rotated
--  into place with os.rename(), which is atomic on POSIX.
--
--  With keep_last = K > 1 the rotation is oldest-first:
--      rm path.(K-1); path.(K-2)->path.(K-1); ... ; path.1->path.2;
--      path->path.1; tmp->path
--  There is a sub-millisecond window between the last two renames where
--  `path` does not exist -- but a COMPLETE, VALID checkpoint exists at
--  path.1 for the whole window, so the invariant "the only copy on disk is
--  never corrupt" holds.  LMTrain.latest_checkpoint() probes that window
--  away for the caller.
--
--  Durability note, stated honestly: rename(2) gives atomic *visibility*.
--  True power-loss durability needs fsync, which pure Lua cannot issue.
--  This protects fully against the reported failure modes (process crash,
--  SIGKILL, OOM-killer, notebook disconnect, forgotten :save()).  A kernel
--  panic mid-write is a weaker guarantee, mitigated by keep_last > 1.

local sformat = string.format
local huge = math.huge
local min = math.min

local ffi_ok, ffi = pcall(require, "ffi")
local HAVE_FFI = ffi_ok and ffi ~= nil
local LITTLE   = HAVE_FFI and ffi.abi("le") or true

local ok_lmlog, lmlog = pcall(require, "lmlog")
local function say(self, text)
    if self and self._reporter then return self._reporter:say(text) end
    if ok_lmlog then return lmlog.say(text) end
    print(text)
end

local FORMAT_TAG = "lanternl-checkpoint"
local FORMAT_VERSION = 3
local CHUNK = 1048576          -- elements per staging pass (4 MB f32 / 8 MB f64)

local function fmt_num(v, prec)
    if v ~= v then return "nan" end
    if v == huge then return "inf" end
    if v == -huge then return "-inf" end
    return sformat(prec, v)
end

local function parse_num(s)
    local n = tonumber(s)
    if n then return n end
    s = string.gsub(string.lower(s), "%s", "")
    if s == "inf" or s == "+inf" then return huge end
    if s == "-inf" then return -huge end
    if s == "nan" or s == "-nan" then return 0 / 0 end
    return nil
end

local function strip_cr(line)
    if line == nil then return nil end
    return (string.gsub(line, "\r$", ""))
end

local function shell(cmd)
    local a, b, c = os.execute(cmd)
    if type(a) == "number" then return a == 0, a end
    if a == true then return true, c or 0 end
    return false, c or b or a
end

local function file_exists(path)
    local f = io.open(path, "rb")
    if not f then return false end
    f:close()
    return true
end

--- Unique-per-process temp suffix.  Deterministic within a run so a crash
--- leaves at most ONE stale tmp file, and distinct across concurrent runs
--- so two processes sharing a path cannot shred each other's partial write.
local TMP_SUFFIX
do
    local seed = tostring(os.time()) .. tostring(math.random(0, 999999))
    if HAVE_FFI then
        local ok = pcall(ffi.cdef, "int getpid(void);")
        if ok then
            local okp, pid = pcall(function() return ffi.C.getpid() end)
            if okp then seed = tostring(tonumber(pid)) .. "-" .. seed end
        end
    end
    TMP_SUFFIX = ".tmp" .. seed
end

--- os.rename() is atomic replace on POSIX.  Windows' MSVCRT rename() fails
--- if the destination exists, so fall back to remove+rename there -- non
--- atomic, and we say so rather than pretending otherwise.
local function rename_over(src, dst)
    local ok, err = os.rename(src, dst)
    if ok then return true end
    if file_exists(dst) then
        os.remove(dst)
        ok, err = os.rename(src, dst)
        if ok then return true end
    end
    return false, err
end

local function rotate(path, keep)
    if not keep or keep <= 1 then return end
    os.remove(path .. "." .. (keep - 1))
    for i = keep - 2, 1, -1 do
        os.rename(path .. "." .. i, path .. "." .. (i + 1))
    end
    os.rename(path, path .. ".1")
end

--- Atomic byte-for-byte copy (used when the "latest" and "best" files are
--- both due in the same tick: serialize once, copy on disk, so there is
--- never more than one D2H pass over the parameters per save point).
local function copy_file_atomic(src, dst)
    local fi, e1 = io.open(src, "rb")
    if not fi then return nil, tostring(e1) end
    local tmp = dst .. TMP_SUFFIX
    local fo, e2 = io.open(tmp, "wb")
    if not fo then fi:close() return nil, tostring(e2) end
    while true do
        local blk = fi:read(4 * 1048576)
        if not blk or #blk == 0 then break end
        if not fo:write(blk) then
            fi:close() fo:close() os.remove(tmp)
            return nil, "short write (disk full?)"
        end
    end
    fi:close()
    local okc, ec = fo:close()
    if okc == false then os.remove(tmp) return nil, tostring(ec) end
    local okr, er = rename_over(tmp, dst)
    if not okr then os.remove(tmp) return nil, tostring(er) end
    return true
end

--- "5" -> 5 epochs ; "30s"/"5m"/"2h" -> wall-clock seconds.
local function parse_interval(v)
    if v == nil then return 1, nil end
    if type(v) == "number" then return v, nil end
    if type(v) == "string" then
        local num, unit = string.match(v, "^%s*(%d+%.?%d*)%s*(%a*)%s*$")
        num = tonumber(num)
        if not num then return nil, nil, "cannot parse" end
        unit = string.lower(unit or "")
        if unit == "" then return num, nil end
        if unit == "s" or unit == "sec" or unit == "secs" then return nil, num end
        if unit == "m" or unit == "min" or unit == "mins" then return nil, num * 60 end
        if unit == "h" or unit == "hr" or unit == "hrs" then return nil, num * 3600 end
        if unit == "e" or unit == "ep" or unit == "epoch" or unit == "epochs" then
            return num, nil
        end
        return nil, nil, "unknown unit '" .. unit .. "'"
    end
    return nil, nil, "must be a number or a string like '5' / '10m'"
end

return function(LMTrain)

    LMTrain.FORMAT_TAG     = FORMAT_TAG
    LMTrain.FORMAT_VERSION = FORMAT_VERSION

    -- ================================================================
    --  Parameter access hooks.
    --
    --  These exist so lmio never has to know which backend it is talking
    --  to, and -- critically -- so the GPU path can stream straight out of
    --  VRAM in fixed-size chunks instead of materialising a 134M-entry Lua
    --  table via _gpu_pull().  lmtrain_gpu.lua overrides all three.
    -- ================================================================

    function LMTrain:_param_dtype() return "f64" end

    --- Copy elements [from, from+count-1] (1-indexed) of parameter i into
    --- the 0-indexed FFI buffer `dst`.
    function LMTrain:_param_get(i, from, count, dst)
        local d = self.params[i].data.data
        for j = 0, count - 1 do dst[j] = d[from + j] end
    end

    --- Inverse of _param_get.
    function LMTrain:_param_set(i, from, count, src)
        local d = self.params[i].data.data
        for j = 0, count - 1 do d[from + j] = src[j] end
    end

    local function alloc_buf(dtype, n)
        return ffi.new(dtype == "f32" and "float[?]" or "double[?]", n)
    end
    local function esize(dtype) return dtype == "f32" and 4 or 8 end

    -- ================================================================
    --  Serialisation
    -- ================================================================

    local function default_precision(dtype)
        -- 9 significant digits round-trip an IEEE binary32 exactly; %.17g
        -- on fp32 data writes 8 bytes per value of pure noise.
        return dtype == "f32" and "%.9g" or "%.17g"
    end

    local function write_header(self, f, fmt, dtype, extra)
        if not f:write(sformat("%s %d\n", FORMAT_TAG, FORMAT_VERSION)) then return nil end
        local parts = {
            sformat("format=%s", fmt),
            sformat("dtype=%s", dtype),
            "endian=le",
            sformat("vocab=%d", self.vocab),
            sformat("dim=%d", self.dim),
            sformat("layers=%d", self.layers),
            sformat("heads=%d", self.heads),
            sformat("ffn=%d", self.ffn),
            sformat("pos=%s", tostring(self.pos or "rope")),
            sformat("causal=%d", (self.causal ~= false) and 1 or 0),
            -- tie changes the PARAMETER COUNT (untied adds a head matrix) and
            -- max_seq / rope_base change behaviour.  None of the three were
            -- recorded in v2, so a tie mismatch surfaced as a baffling
            -- "N tensors vs M tensors" error and a max_seq mismatch was
            -- silently wrong.  Recorded now.
            sformat("tie=%d", (self.tie ~= false) and 1 or 0),
            sformat("max_seq=%d", self.max_seq or 128),
            sformat("rope_base=%g", self.rope_base or 10000),
            sformat("nparams=%d", #self.params),
            sformat("backend=%s", tostring(self.backend or "cpu")),
            sformat("epoch=%d", self.epoch or 0),
            sformat("loss=%s", self.loss and fmt_num(self.loss, "%.9g") or "nan"),
            sformat("best_loss=%s",
                (self.best_loss and self.best_loss < huge)
                    and fmt_num(self.best_loss, "%.9g") or "nan"),
        }
        if extra then parts[#parts + 1] = extra end
        return f:write(table.concat(parts, " "), "\n")
    end

    local function write_params(self, f, fmt, dtype, prec)
        local es = esize(dtype)
        local buf = HAVE_FFI and alloc_buf(dtype, CHUNK) or nil
        local strs = (fmt == "text") and {} or nil

        for i = 1, #self.params do
            local m = self.params[i].data
            local rows, cols = m.rows, m.cols
            local total = rows * cols

            if fmt == "bin" then
                if not f:write(sformat("%d %d\n", rows, cols)) then
                    return nil, "short write"
                end
                local k = 1
                while k <= total do
                    local c = min(CHUNK, total - k + 1)
                    self:_param_get(i, k, c, buf)
                    if not f:write(ffi.string(buf, c * es)) then
                        return nil, "short write (disk full?)"
                    end
                    k = k + c
                end
            else
                if not f:write(sformat("%d %d|", rows, cols)) then
                    return nil, "short write"
                end
                local k = 1
                local first = true
                while k <= total do
                    local c = min(CHUNK, total - k + 1)
                    if buf then
                        self:_param_get(i, k, c, buf)
                        for j = 0, c - 1 do strs[j + 1] = fmt_num(buf[j], prec) end
                    else
                        -- no ffi: read the flat store directly (CPU only)
                        local d = m.data
                        for j = 0, c - 1 do strs[j + 1] = fmt_num(d[k + j], prec) end
                    end
                    for j = c + 1, #strs do strs[j] = nil end
                    local piece = table.concat(strs, ",")
                    if not f:write(first and piece or ("," .. piece)) then
                        return nil, "short write (disk full?)"
                    end
                    first = false
                    k = k + c
                end
                if not f:write("\n") then return nil, "short write" end
            end
        end
        return true
    end

    --- Serialise the whole model to `path` atomically.  Returns ok, err.
    --- Never raises: callers on the training hot path must not die because
    --- a disk filled up.
    function LMTrain:_write_checkpoint(path, opts)
        opts = opts or {}
        if self.params == nil or #self.params == 0 then
            return nil, "no parameters to save (build or train the model first)"
        end

        local dtype = self:_param_dtype()
        local fmt = opts.format or self.save_format
        if fmt == nil then fmt = HAVE_FFI and "bin" or "text" end
        if fmt == "binary" then fmt = "bin" end
        if fmt == "bin" and not HAVE_FFI then fmt = "text" end
        if fmt ~= "bin" and fmt ~= "text" then
            return nil, sformat("unknown save format '%s' (use 'bin' or 'text')", tostring(fmt))
        end
        if fmt == "bin" and not LITTLE then
            return nil, "binary checkpoints assume little-endian; use format = 'text'"
        end
        local prec = opts.precision or self.precision or default_precision(dtype)

        local tmp = path .. TMP_SUFFIX
        local f, err = io.open(tmp, "wb")
        if not f then return nil, sformat("cannot write %s: %s", tmp, tostring(err)) end

        local ok, werr = pcall(function()
            if not write_header(self, f, fmt, dtype, opts.extra_header) then
                error("short write in header", 0)
            end
            local o, e = write_params(self, f, fmt, dtype, prec)
            if not o then error(e or "short write", 0) end
        end)

        if not ok then
            f:close(); os.remove(tmp)
            return nil, tostring(werr)
        end

        -- f:close() flushes.  Its result is the last chance to notice ENOSPC;
        -- the old code discarded it and happily printed "Saved to ...".
        local okc, cerr = f:close()
        if okc == false then
            os.remove(tmp)
            return nil, sformat("close failed (disk full?): %s", tostring(cerr))
        end

        rotate(path, opts.keep or 1)
        local okr, rerr = rename_over(tmp, path)
        if not okr then
            os.remove(tmp)
            return nil, sformat("rename %s -> %s failed: %s", tmp, path, tostring(rerr))
        end
        return true
    end

    -- ================================================================
    --  Public :save / :load
    -- ================================================================

    function LMTrain:save(path, opts)
        if type(path) ~= "string" or path == "" then
            error("LMTrain:save: expected a file path", 2)
        end
        opts = opts or {}
        local ok, err = self:_write_checkpoint(path, opts)
        if not ok then error("LMTrain:save: " .. tostring(err), 2) end
        if self.verbose ~= false and opts.quiet ~= true then
            local fmt = opts.format or self.save_format
                        or (HAVE_FFI and "bin" or "text")
            say(self, sformat("Saved to %s (%s, %s)", path, fmt, self:_param_dtype()))
        end
        return self
    end

    --- Probe `path`, then its rotations, then the best file, and return the
    --- first one whose header parses.  Closes the sub-millisecond rotation
    --- window: after any crash there is always something here to resume from.
    function LMTrain.latest_checkpoint(path, keep)
        if type(path) ~= "string" then return nil end
        local cands = { path }
        for i = 1, (keep or 8) do cands[#cands + 1] = path .. "." .. i end
        cands[#cands + 1] = path .. ".best"
        for _, p in ipairs(cands) do
            local f = io.open(p, "rb")
            if f then
                local l1 = strip_cr(f:read("*l"))
                f:close()
                if l1 and (string.find(l1, FORMAT_TAG, 1, true) == 1
                           or string.match(l1, "vocab=(%d+)")) then
                    return p
                end
            end
        end
        return nil
    end

    function LMTrain:load(path)
        local f, err = io.open(path, "rb")
        if not f then
            error("LMTrain:load: cannot open " .. tostring(path) .. ": " .. tostring(err), 2)
        end

        local line = strip_cr(f:read("*l"))
        if line == nil then
            f:close()
            error("LMTrain:load: " .. path .. " is empty", 2)
        end

        -- ------------------------------------------------------------
        --  CRITICAL FIX.  This used to be:
        --      string.find(line, "^" .. FORMAT_TAG)
        --  FORMAT_TAG is "lanternl-checkpoint", and '-' is a Lua pattern
        --  QUANTIFIER (lazy zero-or-more).  The pattern therefore meant
        --  "lantern" .. (zero or more 'l') .. "checkpoint", which can never
        --  match the literal tag.  Every v2 file fell through to the legacy
        --  branch, tried to parse the tag line as the header, and died with
        --  "unrecognised header" -- i.e. NOTHING :save() ever wrote could be
        --  loaded back.  Plain-text find, anchored explicitly.
        -- ------------------------------------------------------------
        local legacy = true
        local version = 1
        local header = line
        if string.find(line, FORMAT_TAG, 1, true) == 1 then
            legacy = false
            version = tonumber(string.match(line, "^%S+%s+(%d+)")) or 2
            if version > FORMAT_VERSION then
                f:close()
                error(sformat("LMTrain:load: %s is format v%d, this build understands up to v%d "
                    .. "-- upgrade lanternl", path, version, FORMAT_VERSION), 2)
            end
            header = strip_cr(f:read("*l"))
            if header == nil then
                f:close()
                error("LMTrain:load: " .. path .. " is truncated (no header line)", 2)
            end
        end

        local vocab, dim, layers, heads, ffn =
            string.match(header, "vocab=(%d+) dim=(%d+) layers=(%d+) heads=(%d+) ffn=(%d+)")
        if not vocab then
            f:close()
            error("LMTrain:load: unrecognised header in " .. path, 2)
        end

        self.vocab, self.dim, self.layers, self.heads, self.ffn =
            tonumber(vocab), tonumber(dim), tonumber(layers), tonumber(heads), tonumber(ffn)

        local fmt   = string.match(header, "format=(%a+)") or "text"
        local dtype = string.match(header, "dtype=(%w+)") or "f64"
        local endian = string.match(header, "endian=(%a+)")

        if not legacy then
            local pos    = string.match(header, "pos=([%w_]+)")
            local causal = string.match(header, "causal=(%d)")
            local tie    = string.match(header, "tie=(%d)")
            local mseq   = string.match(header, "max_seq=(%d+)")
            local rbase  = string.match(header, "rope_base=([%w%.%+%-]+)")
            if pos then self.pos = pos end
            if causal then self.causal = (causal == "1") end
            if tie then self.tie = (tie == "1") end          -- v3: affects nparams
            if mseq then self.max_seq = tonumber(mseq) end
            if rbase then self.rope_base = tonumber(rbase) or self.rope_base end
            local bl = string.match(header, "best_loss=(%S+)")
            if bl then
                local n = parse_num(bl)
                if n and n == n then self.loaded_best_loss = n end
            end
            local ep = string.match(header, "epoch=(%d+)")
            if ep then self.loaded_epoch = tonumber(ep) end
        end
        -- Legacy checkpoints predate RoPE, but RoPE adds no parameters, so
        -- the weight layout is identical and the file still loads cleanly.

        if fmt == "bin" then
            if not HAVE_FFI then
                f:close()
                error("LMTrain:load: " .. path .. " is a binary checkpoint and this "
                    .. "interpreter has no ffi; re-save with format = 'text' on a LuaJIT box", 2)
            end
            if endian == "be" or not LITTLE then
                f:close()
                error("LMTrain:load: endianness mismatch reading " .. path, 2)
            end
        end

        self._auto_vocab = false
        self:_build()

        local expected = tonumber(string.match(header, "nparams=(%d+)") or "")
        if expected and expected ~= #self.params then
            f:close()
            error(sformat("LMTrain:load: checkpoint has %d parameter tensors but this model "
                .. "has %d (tie/arch mismatch?)", expected, #self.params), 2)
        end

        local model_dtype = self:_param_dtype()
        local fbuf, mbuf
        if HAVE_FFI then
            fbuf = alloc_buf(dtype, CHUNK)
            mbuf = (dtype == model_dtype) and fbuf or alloc_buf(model_dtype, CHUNK)
        end

        for i = 1, #self.params do
            local m = self.params[i].data
            local total = m.rows * m.cols

            if fmt == "bin" then
                local dims = strip_cr(f:read("*l"))
                if dims == nil then
                    f:close()
                    error(sformat("LMTrain:load: %s is truncated at parameter %d", path, i), 2)
                end
                local r, c = string.match(dims, "^(%d+) (%d+)$")
                if not r or tonumber(r) ~= m.rows or tonumber(c) ~= m.cols then
                    f:close()
                    error(sformat("LMTrain:load: parameter %d is %sx%s in the file but %dx%d "
                        .. "in the model", i, tostring(r), tostring(c), m.rows, m.cols), 2)
                end
                local es = esize(dtype)
                local k = 1
                while k <= total do
                    local cn = min(CHUNK, total - k + 1)
                    local want = cn * es
                    local blob = f:read(want)
                    if not blob or #blob ~= want then
                        f:close()
                        error(sformat("LMTrain:load: %s is truncated inside parameter %d", path, i), 2)
                    end
                    ffi.copy(fbuf, blob, want)
                    if mbuf ~= fbuf then
                        for j = 0, cn - 1 do mbuf[j] = fbuf[j] end
                    end
                    self:_param_set(i, k, cn, mbuf)
                    k = k + cn
                end
            else
                local raw = strip_cr(f:read("*l"))
                if raw == nil then
                    f:close()
                    error(sformat("LMTrain:load: %s is truncated at parameter %d", path, i), 2)
                end
                local body = raw
                if not legacy then
                    local r, c, rest = string.match(raw, "^(%d+) (%d+)|(.*)$")
                    if not r then
                        f:close()
                        error(sformat("LMTrain:load: malformed parameter line %d", i), 2)
                    end
                    if tonumber(r) ~= m.rows or tonumber(c) ~= m.cols then
                        f:close()
                        error(sformat("LMTrain:load: parameter %d is %sx%s in the file but "
                            .. "%dx%d in the model", i, r, c, m.rows, m.cols), 2)
                    end
                    body = rest
                end

                local j = 0            -- values seen
                local fill = 0         -- values buffered
                local base = 1
                for num_str in string.gmatch(body, "[^,]+") do
                    j = j + 1
                    if j > total then break end
                    local v = parse_num(num_str)
                    if v == nil then
                        f:close()
                        error(sformat("LMTrain:load: parameter %d value %d is not a number ('%s')",
                            i, j, num_str), 2)
                    end
                    if mbuf then
                        mbuf[fill] = v
                        fill = fill + 1
                        if fill == CHUNK then
                            self:_param_set(i, base, fill, mbuf)
                            base = base + fill
                            fill = 0
                        end
                    else
                        m.data[j] = v
                    end
                end
                if mbuf and fill > 0 then self:_param_set(i, base, fill, mbuf) end
                if j ~= total then
                    f:close()
                    error(sformat("LMTrain:load: parameter %d has %d values, expected %d",
                        i, j, total), 2)
                end
            end
        end

        f:close()
        self._dirty = false
        if self.verbose ~= false then
            say(self, sformat("Loaded from %s (%s v%d, %s)", path, fmt, version, dtype))
        end
        return self
    end

    -- ================================================================
    --  AUTO-SAVE  (PROBLEM 3)
    --
    --  Lives here, not in the two run loops, so `run()` and `_run_gpu()`
    --  call literally the same three methods and cannot drift apart.
    --
    --  Relationship to keep_best -- deliberately orthogonal:
    --    keep_best  : in-memory (host RAM on CPU, GPU-to-GPU on the GPU
    --                 path), fast, volatile, restored at end of run.
    --    auto_save  : on-disk, survives SIGKILL/OOM/disconnect.
    --  Auto-save NEVER calls :snapshot()/:restore().  When `is_best` fires
    --  the live weights already ARE the best, so the .best file is written
    --  straight from them -- no extra GPU-to-GPU copy, no window where the
    --  live model holds stale weights.
    -- ================================================================

    function LMTrain:_autosave_begin()
        self._as = nil
        self.save_error = nil

        local path = self.auto_save
        if path == nil or path == false then return end
        if path == true then
            path = (self.name and tostring(self.name) or "lanternl") .. ".ckpt"
        end
        if type(path) ~= "string" or path == "" then
            error("LMTrain: 'auto_save' must be a file path, true, or false", 2)
        end

        local ep, sec, perr = parse_interval(self.save_every)
        if perr then
            error(sformat("LMTrain: bad 'save_every' (%s): %s",
                tostring(self.save_every), perr), 2)
        end

        local keep = self.keep_last or 1
        local st = {
            path      = path,
            best_path = (self.save_best ~= false) and (path .. ".best") or nil,
            epochs    = ep,
            seconds   = sec,
            keep      = keep,
            last_save = (ok_lmlog and lmlog.now() or os.time()),
            last_epoch = 0,
            writes = 0, fails = 0,
        }

        -- Fail fast.  Discovering a read-only Drive mount or a typo'd
        -- directory at epoch 300 is the same lost run we are trying to
        -- prevent.  Probe uses the per-process suffix so concurrent runs
        -- sharing a path cannot delete each other's probe.
        local probe = path .. TMP_SUFFIX .. ".probe"
        local pf, perr2 = io.open(probe, "wb")
        if not pf then
            error(sformat("LMTrain: auto_save cannot write near '%s': %s\n"
                .. "  (check the directory exists and is writable; on Colab, mount "
                .. "Drive before starting the run)", path, tostring(perr2)), 2)
        end
        pf:write("lanternl")
        local okc = pf:close()
        os.remove(probe)
        if okc == false then
            error(sformat("LMTrain: auto_save probe write to '%s' failed (disk full?)", path), 2)
        end

        self._as = st

        if self.verbose then
            local nparam = 0
            for i = 1, #self.params do
                local m = self.params[i].data
                nparam = nparam + m.rows * m.cols
            end
            local dtype = self:_param_dtype()
            local fmt = self.save_format or (HAVE_FFI and "bin" or "text")
            local per = (fmt == "bin") and esize(dtype)
                        or ((dtype == "f32") and 13 or 20)
            local est = nparam * per
            local when = st.seconds and sformat("every %gs", st.seconds)
                                     or sformat("every %g epoch(s)", st.epochs or 1)
            say(self, sformat(
                "[LMTrain] auto_save -> %s (%s, %s, keep_last=%d, %s, ~%s/checkpoint%s)",
                path, fmt, dtype, keep, when,
                (ok_lmlog and lmlog.fmt_bytes(est) or (est .. "B")),
                st.best_path and (", best -> " .. st.best_path) or ""))
            if est * (keep + (st.best_path and 1 or 0)) > 4 * 1073741824 then
                say(self, sformat("[LMTrain] note: keep_last=%d will occupy ~%s on disk; "
                    .. "lower keep_last or set precision/save_format if space is tight",
                    keep, ok_lmlog and lmlog.fmt_bytes(est * (keep + 1)) or "a lot"))
            end
        end
        return self
    end

    --- kind: "batch" | "epoch" | "final"
    function LMTrain:_autosave_tick(kind)
        local st = self._as
        if not st then return false end
        local t = ok_lmlog and lmlog.now() or os.time()

        local want_latest, reason = false, nil
        if kind == "batch" then
            if st.seconds and (t - st.last_save) >= st.seconds then
                want_latest, reason = true, "time"
            end
        elseif kind == "epoch" then
            if st.epochs and st.epochs > 0
               and (self.epoch - st.last_epoch) >= st.epochs then
                want_latest, reason = true, "epoch"
            elseif st.seconds and (t - st.last_save) >= st.seconds then
                want_latest, reason = true, "time"
            end
        elseif kind == "final" then
            if self.save_at_end == false then
                want_latest = false
            elseif self.stopped == "diverged" then
                -- Writing NaN-poisoned weights over the last good checkpoint
                -- is exactly the loss this feature exists to prevent.
                want_latest = false
                say(self, sformat("[LMTrain] auto_save: run diverged, NOT overwriting %s. "
                    .. "Recover from %s", st.path,
                    st.best_path and (st.best_path .. " or " .. st.path)
                                  or ("the last rotation of " .. st.path)))
            else
                want_latest, reason = true, "final"
            end
        end

        -- Best file only on the epoch the improvement actually happened, and
        -- only when the monitor was genuinely computed this epoch (see the
        -- eval_every staleness fix in the run loops).
        local want_best = (kind == "epoch") and st.best_path
                          and self.is_best and self._monitor_valid
                          and (st.best_epoch ~= self.epoch)

        if not (want_latest or want_best) then return false end

        local ok, err
        if want_latest then
            ok, err = self:_write_checkpoint(st.path, { keep = st.keep })
            if ok then
                st.last_save = t
                st.last_epoch = self.epoch
                st.writes = st.writes + 1
                st.fails = 0
                if want_best then
                    -- One serialisation per tick: copy the bytes we just
                    -- wrote rather than re-reading every parameter again.
                    local okc, ec = copy_file_atomic(st.path, st.best_path)
                    if okc then st.best_epoch = self.epoch
                    else say(self, "[LMTrain] auto_save: best copy failed: " .. tostring(ec)) end
                end
            end
        else
            ok, err = self:_write_checkpoint(st.best_path, { keep = 1 })
            if ok then st.best_epoch = self.epoch end
        end

        if not ok then
            st.fails = st.fails + 1
            self.save_error = tostring(err)
            -- Never abort the run over a failed save: the in-memory model is
            -- still the user's only copy and they may free space.  But shout
            -- every single time -- silence here is the original bug.
            say(self, sformat("[LMTrain] !! auto_save FAILED (attempt %d): %s",
                st.fails, tostring(err)))
            return false
        end

        if self.verbose then
            say(self, sformat("[LMTrain] checkpoint saved: %s (epoch %d, %s)",
                want_latest and st.path or st.best_path, self.epoch or 0,
                reason or "best"))
        end
        return true
    end

    function LMTrain:_autosave_end()
        local st = self._as
        if not st then return end
        if self.save_error and st.fails > 0 then
            say(self, sformat("[LMTrain] WARNING: the last auto_save attempt failed (%s). "
                .. "The most recent checkpoint on disk may be older than the model in "
                .. "memory -- call train:save('somewhere') now.", tostring(self.save_error)))
        elseif self.verbose and st.writes > 0 then
            say(self, sformat("[LMTrain] auto_save wrote %d checkpoint(s); resume with "
                .. "LMTrain{ ..., auto_save = %q, resume = true }", st.writes, st.path))
        end
        self._as = nil
    end

    --- Weights-only resume.  Honest about its limits: optimizer moments and
    --- the LR-schedule position are NOT persisted (Adam m/v are not part of
    --- `params` on either backend), so the schedule restarts from step 0.
    function LMTrain:_maybe_resume()
        local r = self.resume
        if r == nil or r == false then return false end
        local path
        if type(r) == "string" then
            path = LMTrain.latest_checkpoint(r, self.keep_last or 1) or r
        else
            local base = self.auto_save
            if base == true then
                base = (self.name and tostring(self.name) or "lanternl") .. ".ckpt"
            end
            if type(base) ~= "string" then
                error("LMTrain: resume = true needs 'auto_save' set to a path "
                    .. "(or pass resume = 'path/to/ckpt')", 2)
            end
            path = LMTrain.latest_checkpoint(base, self.keep_last or 1)
            if not path then
                if self.verbose then
                    say(self, sformat("[LMTrain] resume: no checkpoint found near %s, "
                        .. "starting fresh", base))
                end
                return false
            end
        end
        self:load(path)
        if self.loaded_best_loss then self.best_loss = self.loaded_best_loss end
        if self.verbose then
            say(self, sformat("[LMTrain] resumed weights from %s (was epoch %s, best %s). "
                .. "Note: optimizer moments and the LR schedule restart from zero.",
                path, tostring(self.loaded_epoch or "?"),
                self.loaded_best_loss and sformat("%.5f", self.loaded_best_loss) or "n/a"))
        end
        return true
    end

    -- ================================================================
    function LMTrain:push(repo_id, opts)
        if type(repo_id) ~= "string" or not string.find(repo_id, "^[%w%-%._]+/[%w%-%._]+$") then
            error("LMTrain:push: repo_id must look like 'user/model', got " .. tostring(repo_id), 2)
        end
        opts = opts or {}
        local file = opts.file or "lanternl_model.bin"
        self:save(file, opts)

        -- Users must run "hf auth login" or "huggingface-cli login" once.
        local cmds = {
            sformat('hf upload %s %s %s', repo_id, file, file),
            sformat('huggingface-cli upload %s %s %s', repo_id, file, file),
        }
        for i = 1, #cmds do
            if shell(cmds[i]) then
                if self.verbose ~= false then
                    say(self, "Pushed " .. file .. " to " .. repo_id)
                end
                return true
            end
        end
        say(self, "LMTrain:push: upload failed -- ensure the 'hf' CLI is installed "
            .. "and you are logged in via `hf auth login`.")
        return false
    end

end
