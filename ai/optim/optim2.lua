local optim2 = {}

local sqrt = math.sqrt
local huge = math.huge

local function param_list(params, who)
    if type(params) ~= "table" or #params == 0 then
        error(who .. ": expected a non-empty list of parameters", 3)
    end
    for i = 1, #params do
        local p = params[i]
        if type(p) ~= "table" or type(p.data) ~= "table" or type(p.grad) ~= "table" then
            error(who .. ": parameter " .. i .. " has no .data / .grad matrix", 3)
        end
    end
    return params
end

function optim2.zero(params)
    for i = 1, #params do
        local g = params[i].grad
        local gd = g.data
        for k = 1, g.rows * g.cols do gd[k] = 0 end
    end
end

function optim2.grad_norm(params)
    local s = 0
    for i = 1, #params do
        local g = params[i].grad
        local gd = g.data
        for k = 1, g.rows * g.cols do
            local v = gd[k]
            s = s + v * v
        end
    end
    return sqrt(s)
end

function optim2.scale_grads(params, f)
    for i = 1, #params do
        local g = params[i].grad
        local gd = g.data
        for k = 1, g.rows * g.cols do gd[k] = gd[k] * f end
    end
end

function optim2.SGD(params, cfg)
    cfg = cfg or {}
    param_list(params, "optim2.SGD")
    local self = {
        name = "sgd",
        params = params,
        lr = cfg.lr or 0.01,
        momentum = cfg.momentum or 0,
        nesterov = cfg.nesterov or false,
        weight_decay = cfg.weight_decay or 0,
    }
    local vel = {}
    if self.momentum > 0 then
        for i = 1, #params do
            local p = params[i]
            local n = p.data.rows * p.data.cols
            local t = {}
            for k = 1, n do t[k] = 0 end
            vel[i] = t
        end
    end

    self.zero_grad = function() optim2.zero(params) end

    self.step = function()
        local lr, mom, wd = self.lr, self.momentum, self.weight_decay
        for i = 1, #params do
            local p = params[i]
            local pd, gd = p.data.data, p.grad.data
            local n = p.data.rows * p.data.cols
            if mom > 0 then
                local v = vel[i]
                for k = 1, n do
                    local gk = gd[k]
                    if wd ~= 0 then gk = gk + wd * pd[k] end
                    local vk = mom * v[k] + gk
                    v[k] = vk
                    if self.nesterov then
                        pd[k] = pd[k] - lr * (gk + mom * vk)
                    else
                        pd[k] = pd[k] - lr * vk
                    end
                end
            else
                for k = 1, n do
                    local gk = gd[k]
                    if wd ~= 0 then gk = gk + wd * pd[k] end
                    pd[k] = pd[k] - lr * gk
                end
            end
        end
    end

    return self
end

function optim2.AdamW(params, cfg)
    cfg = cfg or {}
    param_list(params, "optim2.AdamW")
    local self = {
        name = "adamw",
        params = params,
        lr = cfg.lr or 1e-3,
        beta1 = cfg.beta1 or 0.9,
        beta2 = cfg.beta2 or 0.95,
        eps = cfg.eps or 1e-8,
        weight_decay = cfg.weight_decay or 0.01,
        t = 0,
    }
    local m, v = {}, {}
    for i = 1, #params do
        local p = params[i]
        local n = p.data.rows * p.data.cols
        local mi, vi = {}, {}
        for k = 1, n do mi[k] = 0; vi[k] = 0 end
        m[i], v[i] = mi, vi
    end

    self.zero_grad = function() optim2.zero(params) end

    self.step = function()
        self.t = self.t + 1
        local b1, b2, eps, lr, wd = self.beta1, self.beta2, self.eps, self.lr, self.weight_decay
        local bc1 = 1 - b1 ^ self.t
        local bc2 = 1 - b2 ^ self.t
        local step_size = lr / bc1
        local inv_bc2 = 1 / bc2
        for i = 1, #params do
            local p = params[i]
            local pd, gd = p.data.data, p.grad.data
            local mi, vi = m[i], v[i]
            local n = p.data.rows * p.data.cols
            local decay = (wd ~= 0 and not p.no_decay) and lr * wd or 0
            for k = 1, n do
                local gk = gd[k]
                local mk = b1 * mi[k] + (1 - b1) * gk
                local vk = b2 * vi[k] + (1 - b2) * gk * gk
                mi[k] = mk
                vi[k] = vk
                local denom = sqrt(vk * inv_bc2) + eps
                local pk = pd[k]
                if decay ~= 0 then pk = pk - decay * pk end
                pd[k] = pk - step_size * mk / denom
            end
        end
    end

    self.reset = function()
        self.t = 0
        for i = 1, #params do
            local mi, vi = m[i], v[i]
            for k = 1, #mi do mi[k] = 0; vi[k] = 0 end
        end
    end

    return self
end

function optim2.create(kind, params, cfg)
    kind = string.lower(kind or "adamw")
    if kind == "sgd" then return optim2.SGD(params, cfg) end
    if kind == "adam" or kind == "adamw" then return optim2.AdamW(params, cfg) end
    error("optim2.create: unknown optimizer '" .. tostring(kind) .. "'", 2)
end

return optim2
