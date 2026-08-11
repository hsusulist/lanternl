local Value = require("tensor")
local nn = require("nn")
local optim = require("optim")

local Train = {}
Train.__index = Train

function Train.new(opts)
    opts = opts or {}
    local self = setmetatable({}, Train)

    self.layers    = opts.layers or {4, 8, 3}
    self.lr        = opts.lr or 0.05
    self.epochs    = opts.epochs or 200
    self.log_every = opts.log_every or 20
    self.data      = opts.data or {}
    self.verbose   = (opts.verbose == nil) and true or opts.verbose
    self.loss_fn   = opts.loss_fn or "mse"
    self.on_log    = opts.on_log or nil

    self.loss_history = {}
    self.best_loss = math.huge

    self.model  = nn.MLP(self.layers)
    self.params = self.model:parameters()
    self.optim  = optim.SGD(self.params, self.lr)

    return self
end

function Train:config(opts)
    opts = opts or {}
    for key, val in pairs(opts) do
        self[key] = val
    end

    if opts.layers then
        self.model  = nn.MLP(self.layers)
        self.params = self.model:parameters()
    end
    if opts.lr or opts.layers then
        self.optim = optim.SGD(self.params, self.lr)
    end

    return self
end

local function mse_loss(out, target)
    local loss = Value.new(0)
    for i, target_val in ipairs(target) do
        local diff = out[i] + Value.new(-target_val)
        loss = loss + diff ^ 2
    end
    return loss
end

-- Plain text message (headers, info lines)
function Train:log(msg)
    if self.verbose then print(msg) end
end

-- Default loss logger for beginners: epoch, loss, progress bar, best marker
local function default_on_log(info)
    local bar_len = 20
    local filled = math.floor(info.progress * bar_len)
    local bar = string.rep("#", filled) .. string.rep("-", bar_len - filled)

    local marker = info.is_best and " (best)" or ""
    print(string.format(
        "[%s] Epoch %d/%d | Loss: %.5f%s",
        bar, info.epoch, info.total_epochs, info.loss, marker
    ))
end

-- Called every epoch with structured info; uses custom on_log if user set one
function Train:report(epoch, loss)
    table.insert(self.loss_history, loss)

    local is_best = loss < self.best_loss
    if is_best then self.best_loss = loss end

    if not self.verbose then return end

    local info = {
        epoch        = epoch,
        total_epochs = self.epochs,
        loss         = loss,
        progress     = epoch / self.epochs,
        is_best      = is_best,
        history      = self.loss_history,
    }

    if self.on_log then
        self.on_log(info)
    else
        default_on_log(info)
    end
end

function Train:run()
    assert(#self.data > 0, "train: no data set, use train:config{data = {...}}")

    self:log("Model layers: " .. table.concat(self.layers, " -> "))
    self:log("Total parameters: " .. #self.params)

    for epoch = 1, self.epochs do
        local total_loss = Value.new(0)

        for _, sample in ipairs(self.data) do
            local x = {}
            for i, v in ipairs(sample.input) do
                x[i] = Value.new(v)
            end

            local out = self.model:call(x)
            local loss = mse_loss(out, sample.target)
            total_loss = total_loss + loss
        end

        self.optim.zero_grad()
        total_loss:backward()
        self.optim.step()

        if self.log_every and epoch % self.log_every == 0 then
            self:report(epoch, total_loss.data)
        end
    end

    return self.model
end

function Train:evaluate(data)
    data = data or self.data
    for _, sample in ipairs(data) do
        local x = {}
        for i, v in ipairs(sample.input) do
            x[i] = Value.new(v)
        end
        local out = self.model:call(x)

        io.write("Input: ")
        for _, v in ipairs(sample.input) do io.write(v .. " ") end
        io.write("-> Output: ")
        for _, o in ipairs(out) do io.write(string.format("%.3f ", o.data)) end
        print()
    end
end

setmetatable(Train, {
    __call = function(_, ...) return Train.new(...) end
})

return Train