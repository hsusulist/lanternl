local Data = {}
Data.__index = Data

local function file_exists(path)
    local f = io.open(path, "r")
    if f ~= nil then io.close(f) return true else return false end
end

local function format_bytes(bytes)
    if bytes >= 1073741824 then
        return string.format("%.2f GB", bytes / 1073741824)
    elseif bytes >= 1048576 then
        return string.format("%.2f MB", bytes / 1048576)
    elseif bytes >= 1024 then
        return string.format("%.2f KB", bytes / 1024)
    else
        return bytes .. " Bytes"
    end
end

local function get_remote_info(url)
    local handle = io.popen(string.format('curl -s -I -L "%s"', url))
    if not handle then return 0, 0 end
    local response = handle:read("*a")
    handle:close()

    local bytes = 0
    local cl = response:match("[Cc]ontent%-[Ll]ength:%s*(%d+)")
    if cl then bytes = tonumber(cl) or 0 end

    local sample_handle = io.popen(string.format('curl -s -L -r 0-102400 "%s"', url))
    local sample_pairs = 0
    if sample_handle then
        local sample_text = sample_handle:read("*a")
        sample_handle:close()
        for _ in sample_text:gmatch("\n") do
            sample_pairs = sample_pairs + 1
        end
    end

    local estimated_pairs = 0
    if bytes > 0 and sample_pairs > 0 then
        estimated_pairs = math.floor((bytes / 102400) * sample_pairs)
    else
        estimated_pairs = sample_pairs
    end

    return bytes, estimated_pairs
end

local function pull_from_hf(repo_id, limit, filename)
    filename = filename or "train.txt"
    local local_file = repo_id:gsub("/", "_") .. ".txt"

    if file_exists(local_file) then
        return local_file
    end

    local max_bytes = nil
    local max_lines = nil

    if limit then
        local lim_str = tostring(limit):lower():gsub("%s+", "")
        if lim_str:find("gb") then
            local num = tonumber(lim_str:match("(%d+%.?%d*)gb"))
            if num then max_bytes = math.floor(num * 1073741824) end
        elseif lim_str:find("mb") then
            local num = tonumber(lim_str:match("(%d+%.?%d*)mb"))
            if num then max_bytes = math.floor(num * 1048576) end
        elseif lim_str:find("kb") then
            local num = tonumber(lim_str:match("(%d+%.?%d*)kb"))
            if num then max_bytes = math.floor(num * 1024) end
        elseif lim_str:find("b") and not lim_str:find("pair") then
            local num = tonumber(lim_str:match("(%d+%.?%d*)b"))
            if num then max_bytes = math.floor(num) end
        elseif lim_str:find("pair") or lim_str:find("line") then
            local num = tonumber(lim_str:match("(%d+)"))
            if num then max_lines = num end
        end
    end

    local url = string.format("https://huggingface.co/datasets/%s/raw/main/%s", repo_id, filename)

    if not limit then
        print("[ai.Data] Checking dataset size from Hugging Face...")
        local total_bytes, est_pairs = get_remote_info(url)
        local size_str = format_bytes(total_bytes)

        if est_pairs > 10000 then
            print(string.format("Warning: This dataset is large (~%s, ~%d pairs).", size_str, est_pairs))
            io.write("Are you sure you want to download it? (y/n): ")
            io.flush()
            local answer = string.lower(io.read() or "")
            if answer ~= "y" and answer ~= "yes" then
                print("[ai.Data] Download cancelled by user.")
                return nil
            end
        end
    end

    print("[ai.Data] Downloading " .. repo_id .. "...")

    local cmd
    if max_bytes then
        cmd = string.format('curl -L -s -r 0-%d -o "%s" "%s"', max_bytes - 1, local_file, url)
    else
        cmd = string.format('curl -L -s -o "%s" "%s"', local_file, url)
    end

    local ok = os.execute(cmd)

    if ok and max_lines then
        local lines = {}
        local f = io.open(local_file, "r")
        if f then
            for line in f:lines() do
                table.insert(lines, line)
                if #lines >= max_lines then break end
            end
            f:close()

            local f_out = io.open(local_file, "w")
            for _, l in ipairs(lines) do f_out:write(l .. "\n") end
            f_out:close()
        end
    end

    return local_file
end

function Data.new(...)
    local args = {...}
    local self = setmetatable({}, Data)

    self.files = {}
    self.batch_size = 8
    self.shuffle = true
    self.stream = false
    self.tokenizer = nil  

    local sources = {}
    local limit = nil

    for _, arg in ipairs(args) do
        if type(arg) == "string" then
            local lower_arg = arg:lower()
            if lower_arg:find("gb") or lower_arg:find("mb") or lower_arg:find("kb") or lower_arg:find("pair") then
                limit = arg
            else
                table.insert(sources, arg)
            end
        end
    end

    for _, src in ipairs(sources) do
        if src:find("/") and not file_exists(src) then
            local downloaded_path = pull_from_hf(src, limit)
            if downloaded_path then
                table.insert(self.files, downloaded_path)
            end
        else
            table.insert(self.files, src)
        end
    end

    return self
end

function Data:config(opts)
    opts = opts or {}
    self.batch_size = opts.batch_size or self.batch_size
    self.shuffle    = opts.shuffle    ~= nil and opts.shuffle or self.shuffle
    self.stream     = opts.stream     ~= nil and opts.stream or self.stream
    self.tokenizer  = opts.tokenizer  or self.tokenizer
    return self
end

setmetatable(Data, {
    __call = function(_, ...) return Data.new(...) end
})

local function read_all_lines(files)
    local lines = {}
    for _, file in ipairs(files) do
        local f = io.open(file, "r")
        if f then
            for line in f:lines() do
                if line ~= "" then
                    table.insert(lines, line)
                end
            end
            f:close()
        end
    end
    return lines
end

local function shuffle_lines(lines)
    for i = #lines, 2, -1 do
        local j = math.random(i)
        lines[i], lines[j] = lines[j], lines[i]
    end
    return lines
end

function Data:batches()
    local lines = read_all_lines(self.files)

    if self.shuffle then
        lines = shuffle_lines(lines)
    end

    if self.tokenizer then
        for i, line in ipairs(lines) do
            lines[i] = self.tokenizer:encode(line)
        end
    end

    local batches = {}
    local current = {}
    for i, item in ipairs(lines) do
        table.insert(current, item)
        if #current >= self.batch_size then
            table.insert(batches, current)
            current = {}
        end
    end
    if #current > 0 then
        table.insert(batches, current)
    end

    return batches
end

function Data:count()
    return #read_all_lines(self.files)
end

return Data