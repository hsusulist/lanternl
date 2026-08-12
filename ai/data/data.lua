local Data = {}
Data.__index = Data

local function file_exists(path)
    local f = io.open(path, "r")
    if f ~= nil then 
        local size = f:seek("end")
        io.close(f)
        return size > 0
    end
    return false
end

local function format_bytes(bytes)
    if bytes >= 1073741824 then return string.format("%.2f GB", bytes / 1073741824)
    elseif bytes >= 1048576 then return string.format("%.2f MB", bytes / 1048576)
    elseif bytes >= 1024 then return string.format("%.2f KB", bytes / 1024)
    else return bytes .. " Bytes" end
end

-- NEW: Beautiful real-time progress bar!
local function print_progress(current, total)
    local bar_len = 30
    local percent = total > 0 and (current / total) or 0
    if percent > 1 then percent = 1 end
    
    local filled = math.floor(percent * bar_len)
    local bar = string.rep("=", filled) .. string.rep(" ", bar_len - filled)
    
    local cur_str = format_bytes(current)
    local tot_str = format_bytes(total)
    
    io.write(string.format("\r[ai.Data] Downloading [%s] %d%% (%s / %s)", 
        bar, math.floor(percent * 100), cur_str, tot_str))
    io.flush()
end

-- NEW: Downloads in 1MB chunks so we can update the progress bar!
local function download_with_progress(url, local_file, max_bytes)
    local chunk_size = 1048576 -- 1 MB
    local current_bytes = 0
    local f_out = io.open(local_file, "wb")
    if not f_out then return false end

    while true do
        if max_bytes and current_bytes >= max_bytes then break end
        
        local end_byte = current_bytes + chunk_size - 1
        if max_bytes and end_byte >= max_bytes then
            end_byte = max_bytes - 1
        end

        local temp_file = "temp_chunk.bin"
        -- curl -r 0-1048576 gets bytes 0 to 1MB
        local cmd = string.format('curl -s -L -r %d-%d -o "%s" "%s"', current_bytes, end_byte, temp_file, url)
        os.execute(cmd)

        local f_in = io.open(temp_file, "rb")
        if not f_in then break end
        
        local data = f_in:read("*a")
        f_in:close()
        os.remove(temp_file)

        if not data or #data == 0 then break end -- End of file

        f_out:write(data)
        current_bytes = current_bytes + #data

        if max_bytes then
            print_progress(current_bytes, max_bytes)
        else
            -- If no limit, just show how much we've downloaded
            io.write("\r[ai.Data] Downloaded " .. format_bytes(current_bytes) .. "...")
            io.flush()
        end

        -- If we got less data than we asked for, the file is finished
        if #data < chunk_size then break end
    end
    
    f_out:close()
    print(" ") -- Print newline when done
    return current_bytes > 0
end

local function get_hf_file_list(repo_id)
    local url = string.format("https://huggingface.co/api/datasets/%s", repo_id)
    local handle = io.popen(string.format('curl -s -L "%s"', url))
    if not handle then return {} end
    local response = handle:read("*a")
    handle:close()

    local files = {}
    for file_path in response:gmatch('"rfilename"%s*:%s*"([^"]+)"') do
        if file_path:match("%.txt$") or file_path:match("%.csv$") or 
           file_path:match("%.jsonl$") or file_path:match("%.parquet") then
            table.insert(files, file_path)
        end
    end
    return files
end

local function pull_from_hf(repo_id, limit, filename)
    local files_to_download = {}
    if filename then
        files_to_download = { filename }
    else
        print("[ai.Data] Scanning HuggingFace repo '" .. repo_id .. "' for data files...")
        files_to_download = get_hf_file_list(repo_id)
        if #files_to_download == 0 then
            print("[ai.Data] ERROR: No .txt, .csv, .jsonl, or .parquet files found in this repo!")
            return nil
        end
    end

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

    for _, hf_path in ipairs(files_to_download) do
        if max_bytes and total_bytes_downloaded >= max_bytes then
            print("[ai.Data] Reached size limit (" .. limit .. "). Stopping download.")
            break
        end

        local local_file = repo_id:gsub("/", "_") .. "_" .. hf_path:gsub("/", "_")

        if not file_exists(local_file) then
            local url = string.format("https://huggingface.co/datasets/%s/resolve/main/%s", repo_id, hf_path)
            print("\n[ai.Data] File: " .. hf_path)
            
            local remaining_bytes = max_bytes and (max_bytes - total_bytes_downloaded) or nil
            
            local success = download_with_progress(url, local_file, remaining_bytes)
            
            if success and file_exists(local_file) then
                local f = io.open(local_file, "r")
                local first_line = f:read("*l") or ""
                f:close()
                
                if first_line:find("Entry not found") or first_line:find("<!DOCTYPE html>") then
                    print("[ai.Data] WARNING: Failed to download " .. hf_path .. " (404 Not Found)")
                    os.remove(local_file)
                else
                    local size = io.open(local_file, "r"):seek("end")
                    io.close(io.open(local_file, "r"))
                    total_bytes_downloaded = total_bytes_downloaded + size
                    table.insert(downloaded_files, local_file)
                end
            end
        else
            local size = io.open(local_file, "r"):seek("end")
            io.close(io.open(local_file, "r"))
            total_bytes_downloaded = total_bytes_downloaded + size
            table.insert(downloaded_files, local_file)
            print("[ai.Data] Skipping " .. hf_path .. " (already downloaded)")
        end
    end

    if #downloaded_files == 0 then
        print("[ai.Data] ERROR: Failed to download any files.")
        return nil
    end

    return downloaded_files
end

-- NEW: Pure Lua Parquet/Binary text extractor!
-- Parquet is compressed binary, but the actual text strings are often stored as plain bytes inside.
local function extract_text_from_binary(file)
    local f = io.open(file, "rb")
    if not f then return {} end
    local data = f:read("*a")
    f:close()

    local lines = {}
    local current_word = {}
    
    -- Scan every single byte in the file
    for i = 1, #data do
        local byte = data:byte(i)
        
        -- If it's a standard ASCII letter, number, or punctuation, keep it
        if (byte >= 32 and byte <= 126) or byte == 9 then -- 9 is tab
            table.insert(current_word, string.char(byte))
        else
            -- If we hit a binary byte (0x00, compression markers, etc), end the current word
            if #current_word > 0 then
                local word = table.concat(current_word)
                -- Only save it if it looks like a real word (length > 2)
                if #word > 2 and not word:match("^[\x00-\xff]+$") then 
                    -- We join words with spaces so they form lines
                    if lines[#lines] then
                        lines[#lines] = lines[#lines] .. " " .. word
                    else
                        table.insert(lines, word)
                    end
                end
                current_word = {}
            end
            
            -- Newlines in binary often mean end of a row
            if byte == 10 or byte == 13 then -- \n or \r
                if lines[#lines] and #lines[#lines] > 10 then
                    table.insert(lines, "") -- Start a new line
                end
            end
        end
    end
    
    -- Clean up empty lines
    local clean_lines = {}
    for _, l in ipairs(lines) do
        if l and #l > 0 then
            table.insert(clean_lines, l)
        end
    end
    
    return clean_lines
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
        -- Check if the file is a Parquet file
        if file:match("%.parquet") then
            print("[ai.Data] Parsing Parquet binary: " .. file .. " ...")
            local extracted = extract_text_from_binary(file)
            for _, l in ipairs(extracted) do
                table.insert(lines, l)
            end
        else
            -- Normal text file reading
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