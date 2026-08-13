-- lmio.lua -- checkpoint save/load and Hugging Face upload for LMTrain.
-- Usage (from lmtrain.lua):  require("lmio")(LMTrain)

local sformat = string.format
local huge = math.huge

local FORMAT_TAG = "lanternl-checkpoint"
local FORMAT_VERSION = 2

-- "%.17g" round-trips an IEEE double exactly; tostring() uses %.14g and
-- silently truncated ~3 digits off every weight on each save/load cycle.
local function fmt_num(v, prec)
    if v ~= v then return "nan" end
    if v == huge then return "inf" end
    if v == -huge then return "-inf" end
    return sformat(prec, v)
end

local function parse_num(s)
    local n = tonumber(s)
    if n then return n end
    s = string.lower(s)
    s = string.gsub(s, "%s", "")
    if s == "inf" or s == "+inf" then return huge end
    if s == "-inf" then return -huge end
    if s == "nan" or s == "-nan" then return 0 / 0 end
    return nil
end

local function strip_cr(line)
    if line == nil then return nil end
    return (string.gsub(line, "\r$", ""))
end

-- os.execute returns a number on Lua 5.1/LuaJIT (0 == success, but every
-- number is truthy) and (boolean, string, number) on 5.3+. The old
-- `if not ok` test could therefore never detect a failed upload.
local function shell(cmd)
    local a, b, c = os.execute(cmd)
    if type(a) == "number" then return a == 0, a end
    if a == true then return true, c or 0 end
    return false, c or b or a
end

return function(LMTrain)

    function LMTrain:save(path, opts)
        if type(path) ~= "string" or path == "" then
            error("LMTrain:save: expected a file path", 2)
        end
        opts = opts or {}
        local prec = opts.precision or "%.17g"
        if self.params == nil or #self.params == 0 then
            error("LMTrain:save: no parameters to save (build or train the model first)", 2)
        end

        local f, err = io.open(path, "wb")
        if not f then
            error("LMTrain:save: cannot write " .. path .. ": " .. tostring(err), 2)
        end

        f:write(sformat("%s %d\n", FORMAT_TAG, FORMAT_VERSION))
        f:write(sformat("vocab=%d dim=%d layers=%d heads=%d ffn=%d pos=%s causal=%d nparams=%d\n",
            self.vocab, self.dim, self.layers, self.heads, self.ffn,
            tostring(self.pos or "rope"),
            (self.causal ~= false) and 1 or 0,
            #self.params))

        for i = 1, #self.params do
            local m = self.params[i].data
            local d = m.data
            local total = m.rows * m.cols
            local parts = {}
            for k = 1, total do parts[k] = fmt_num(d[k], prec) end
            f:write(sformat("%d %d|", m.rows, m.cols))
            f:write(table.concat(parts, ","))
            f:write("\n")
        end

        f:close()
        if self.verbose ~= false then print("Saved to " .. path) end
        return self
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

        local legacy = true
        local header = line
        if string.find(line, "^" .. FORMAT_TAG) then
            legacy = false
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

        if not legacy then
            local pos = string.match(header, "pos=([%w_]+)")
            local causal = string.match(header, "causal=(%d)")
            if pos then self.pos = pos end
            if causal then self.causal = (causal == "1") end
        end
        -- Legacy checkpoints predate RoPE, but RoPE adds no parameters, so
        -- the weight layout is identical and the file still loads cleanly;
        -- the model simply gains positions and masking it was trained without.

        self._auto_vocab = false
        self:_build()

        local expected = string.match(header, "nparams=(%d+)")
        expected = expected and tonumber(expected) or nil
        if expected and expected ~= #self.params then
            f:close()
            error(sformat("LMTrain:load: checkpoint has %d parameter tensors but this model has %d",
                expected, #self.params), 2)
        end

        for i = 1, #self.params do
            local raw = strip_cr(f:read("*l"))
            if raw == nil then
                f:close()
                error(sformat("LMTrain:load: %s is truncated at parameter %d", path, i), 2)
            end

            local m = self.params[i].data
            local body = raw
            if not legacy then
                local r, c, rest = string.match(raw, "^(%d+) (%d+)|(.*)$")
                if not r then
                    f:close()
                    error(sformat("LMTrain:load: malformed parameter line %d", i), 2)
                end
                if tonumber(r) ~= m.rows or tonumber(c) ~= m.cols then
                    f:close()
                    error(sformat("LMTrain:load: parameter %d is %sx%s in the file but %dx%d in the model",
                        i, r, c, m.rows, m.cols), 2)
                end
                body = rest
            end

            local d, j = m.data, 0
            local total = m.rows * m.cols
            for num_str in string.gmatch(body, "[^,]+") do
                j = j + 1
                if j > total then break end
                local v = parse_num(num_str)
                if v == nil then
                    f:close()
                    error(sformat("LMTrain:load: parameter %d value %d is not a number ('%s')",
                        i, j, num_str), 2)
                end
                d[j] = v
            end
            if j ~= total then
                f:close()
                error(sformat("LMTrain:load: parameter %d has %d values, expected %d", i, j, total), 2)
            end
        end

        f:close()
        self._dirty = false
        if self.verbose ~= false then print("Loaded from " .. path) end
        return self
    end

    function LMTrain:push(repo_id, token, opts)
        if type(repo_id) ~= "string" or not string.find(repo_id, "^[%w%-%._]+/[%w%-%._]+$") then
            error("LMTrain:push: repo_id must look like 'user/model', got " .. tostring(repo_id), 2)
        end
        opts = opts or {}
        local file = opts.file or "lanternl_model.txt"
        self:save(file, opts)

        -- Note: a token passed here lands in the OS process list. Prefer
        -- `hf auth login` (or `huggingface-cli login`) once, then omit it.
        local suffix = token and (" --token " .. token) or ""
        local cmds = {
            sformat('hf upload %s %s %s%s', repo_id, file, file, suffix),
            sformat('huggingface-cli upload %s %s %s%s', repo_id, file, file, suffix),
        }

        for i = 1, #cmds do
            local ok = shell(cmds[i])
            if ok then
                if self.verbose ~= false then
                    print("Pushed " .. file .. " to " .. repo_id)
                end
                return true
            end
        end

        print("LMTrain:push: upload failed -- check that the 'hf' CLI is installed "
            .. "(pip install -U huggingface_hub) and that you are logged in.")
        return false
    end

end
