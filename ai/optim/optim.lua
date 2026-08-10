local optim = {}

function optim.SGD(parameters, lr)
    local self = {}
    self.parameters = parameters
    self.lr = lr or 0.01

    function self.step()
        for _, p in ipairs(self.parameters) do
            p.data = p.data - self.lr * p.grad
        end
    end

    function self.zero_grad()
        for _, p in ipairs(self.parameters) do
            p.grad = 0
        end
    end

    return self
end

return optim