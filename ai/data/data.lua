local Data = {}
Data.__index = Data

-- FIX 1: Check if file exists AND has data in it
local function file_exists(path)
    local f = io.open(path, "r")
    if f ~= nil then 
        local size = f:seek("end") -- Check size
        io.close(f)
        return size > 0 -- Only return true if it's not empty
    end
    return false
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

-- Helper to ask HuggingFace API for all files in a repo
local function get_hf_file_list(repo_id)
    local url = string.format("https://huggingface.co/api/datasets/%s", repo_id)
    local handle = io.popen(string.format('curl -s -L "%s"', url))
    if not handle then return {} end
    local response = handle:read("*a")
    handle:close()

    -- HuggingFace API returns JSON. We use a simple pattern to find all "rfilename": "..."
    local files = {}
    for file_path in response:gmatch('"rfilename"%s*:%s*"([^"]+)"') do
        -- We only care about actual data files
        if file_path:match("%.txt$") or file_path:match("%.csv$") or 
           file_path:match("%.jsonl$") or file_path:match("%.parquet") then
            table.insert(files, file_path)
        end
    end
    return files
end

local function pull_from_hf(repo_id, limit, filename)
    -- 1. Figure out what files we need to download
    local files_to_download = {}
    if filename then
        -- User specified an exact file
        files_to_download = { filename }
    else
        print("[ai.Data] Scanning HuggingFace repo '" .. repo_id .. "' for data files...")
        files_to_download = get_hf_file_list(repo_id)
        if #files_to_download == 0 then
            print("[ai.Data] ERROR: No .txt, .csv, .jsonl, or .parquet files found in this repo!")
            return nil
        end
    end

    -- 2. Parse the user's size limit (e.g., "500MB" -> 524288000 bytes)
    local max_bytes = nil
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
        end
    end

    local downloaded_files = {}
    local total_bytes_downloaded = 0

    -- 3. Download files one by one
    for _, hf_path in ipairs(files_to_download) do
        -- Stop if we hit the user's byte limit
        if max_bytes and total_bytes_downloaded >= max_bytes then
            print("[ai.Data] Reached size limit (" .. limit .. "). Stopping download.")
            break
        end

        -- Create a safe local filename (replace slashes with underscores)
        local local_file = repo_id:gsub("/", "_") .. "_" .. hf_path:gsub("/", "_")

        if not file_exists(local_file) then
            local url = string.format("https://huggingface.co/datasets/%s/resolve/main/%s", repo_id, hf_path)
            print("[ai.Data] Downloading: " .. hf_path)

            local cmd
            if max_bytes then
                -- If there's a limit, only download the remaining bytes we are allowed
                local remaining = max_bytes - total_bytes_downloaded
                cmd = string.format('curl -L -s -r 0-%d -o "%s" "%s"', remaining - 1, local_file, url)
            else
                cmd = string.format('curl -L -s -o "%s" "%s"', local_file, url)
            end

            os.execute(cmd)

            -- Verify it downloaded and isn't an HTML error page
            if file_exists(local_file) then
                local f = io.open(local_file, "r")
                local first_line = f:read("*l") or ""
                f:close()
                
                if first_line:find("Entry not found") or first_line:find("<!DOCTYPE html>") then
                    print("[ai.Data] WARNING: Failed to download " .. hf_path .. " (404 Not Found)")
                    os.remove(local_file)
                else
                    local size = io.open(local_file, "r"):seek("end")
                    io.close(io.open(local_file, "r")) -- close it safely
                    total_bytes_downloaded = total_bytes_downloaded + size
                    table.insert(downloaded_files, local_file)
                end
            end
        else
            -- File already exists locally, skip downloading
            local size = io.open(local_file, "r"):seek("end")
            io.close(io.open(local_file, "r"))
            total_bytes_downloaded = total_bytes_downloaded + size
            table.insert(downloaded_files, local_file)
        end
    end

    if #downloadloaded_files == 0 then
        print("[ai.Data] ERROR: Failed to download any files.")
        return nil
    end

    return downloaded_files -- Returns a TABLE of file paths!
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
            -- Split repo ID and exact filename (if provided)
            local repo_id, filename = src, nil
            local first_slash = src:find("/")
            if first_slash then
                local second_slash = src:find("/", first_slash + 1)
                if second_slash then
                    repo_id = src:sub(1, second_slash - 1)
                    filename = src:sub(second_slash + 1)
                end
            end
            
            local downloaded_paths = pull_from_hf(repo_id, limit, filename)
            if downloaded_paths and type(downloaded_paths) == "table" then
                -- Merge the downloaded files into self.files
                for _, f in ipairs(downloaded_paths) do
                    table.insert(self.files, f)
                end
            else
                error("[ai.Data] Failed to load dataset: " .. src, 2)
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