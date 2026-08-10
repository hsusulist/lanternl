local Value = require("tensor")

local nn = {}

local Neuron = {}
Neuron.__index = Neuron

function nn.Neuron(nin, nonlin)
    local self = setmetatable({}, Neuron)
    self.w = {}
    for i = 1, nin do
        self.w[i] = Value.new(math.random() * 2 - 1)
    end
    self.b = Value.new(0)
    self.nonlin = (nonlin == nil) and true or nonlin
    return self
end

function Neuron:call(x)
    local act = self.b
    for i = 1, #self.w do
        act = act + self.w[i] * x[i]
    end
    if self.nonlin then
        return Value.relu(act)
    else
        return act
    end
end

function Neuron:parameters()
    local params = {}
    for i = 1, #self.w do
        table.insert(params, self.w[i])
    end
    table.insert(params, self.b)
    return params
end

local Layer = {}
Layer.__index = Layer

function nn.Layer(nin, nout, nonlin)
    local self = setmetatable({}, Layer)
    self.neurons = {}
    for i = 1, nout do
        self.neurons[i] = nn.Neuron(nin, nonlin)
    end
    return self
end

function Layer:call(x)
    local out = {}
    for i = 1, #self.neurons do
        out[i] = self.neurons[i]:call(x)
    end
    return out
end

function Layer:parameters()
    local params = {}
    for i = 1, #self.neurons do
        for _, p in ipairs(self.neurons[i]:parameters()) do
            table.insert(params, p)
        end
    end
    return params
end

local MLP = {}
MLP.__index = MLP

-- Last layer automatically set to linear (no ReLU) to avoid dying-ReLU on outputs
function nn.MLP(sizes)
    local self = setmetatable({}, MLP)
    self.layers = {}
    for i = 1, #sizes - 1 do
        local is_last = (i == #sizes - 1)
        self.layers[i] = nn.Layer(sizes[i], sizes[i + 1], not is_last)
    end
    return self
end

function MLP:call(x)
    local out = x
    for i = 1, #self.layers do
        out = self.layers[i]:call(out)
    end
    return out
end

function MLP:parameters()
    local params = {}
    for i = 1, #self.layers do
        for _, p in ipairs(self.layers[i]:parameters()) do
            table.insert(params, p)
        end
    end
    return params
end

nn.Neuron_meta = Neuron
nn.Layer_meta = Layer
nn.MLP_meta = MLP

return nn