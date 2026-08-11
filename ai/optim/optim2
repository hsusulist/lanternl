local optim2 = {}

local DEFAULTS = {
    lr = 0.01,
}

function optim2.SGD(parameters, config)
    config = config or {}
    local self = {}
    self.parameters = parameters
    self.lr = config.lr or DEFAULTS.lr

    function self.step()
        for _, p in ipairs(self.parameters) do
            local total = p.data.rows * p.data.cols
            for k = 1, total do
                p.data.data[k] = p.data.data[k] - self.lr * p.grad.data[k]
            end
        end
    end

    function self.zero_grad()
        for _, p in ipairs(self.parameters) do
            local total = p.grad.rows * p.grad.cols
            for k = 1, total do
                p.grad.data[k] = 0
            end
        end
    end

    return self
end

return optim2
