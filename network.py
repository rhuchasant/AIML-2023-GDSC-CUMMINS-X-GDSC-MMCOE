import random
import numpy as np
from scalarflow.engine import value

class module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.gradient = 0
            
    def parameters(self):
        return []

class neuron(module):
    
    def __init__(self, num_in, activation_func):
        self.weights = [value(random.uniform(-1, 1), label=f'w{i}') for i in range(num_in)]
        self.bias = value(0, label='b')
        self.activation_func = activation_func
    
    def __call__(self, x):
        raw = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        act = self.activation_func(raw)
        return act
    
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f"\nneuron(weights: {len(self.weights)}) - activation: '{self.activation_func.__name__.capitalize()}'"

class layer(module):
    
    def __init__(self, num_in, num_out, activation_func):
        self.neurons = [neuron(num_in, activation_func) for _ in range(num_out)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        params = []
        for n in self.neurons:
            p_unit = n.parameters()
            params.extend(p_unit)
        return params
    
    def __repr__(self):
        return f"\nlayer of {len(self.neurons)} neurons: [{', '.join(str(n) for n in self.neurons)}]"

class network(module):
    
    def __init__(self, num_in, layer_sizes, activation_funcs):
        self.layer_activations = activation_funcs
        size = [num_in] + layer_sizes
        self.layers = [layer(size[i], size[i+1], activation_funcs[i]) for i in range(len(layer_sizes)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            p_layer = layer.parameters()
            params.extend(p_layer)
        return params
    
    def __repr__(self):
        return f"neural network of {len(self.layers)} layers: [{', '.join(str(layer) for layer in self.layers)}\n]"
    
    def summary(self):
        print(f"Network architecture:")
        print("--------------------------------------")
        print(self)
        print("--------------------------------------")
        print(f"Total number of parameters: {len(self.parameters())}")
        print(f"Total number of dense layers: {len(self.layers)}")
        print(f"Total number of neurons: {sum(len(layer.neurons) for layer in self.layers)}")
        print(f"Total number of weights: {sum(len(layer.neurons[0].weights) for layer in self.layers)}")
        print(f"Total number of biases: {sum(len(layer.neurons) for layer in self.layers)}        
        
    # Rest of the code remains the same
    
# Define activation functions
def relu(x):
    return x.relu()

def tanh(x):
    return x.tanh()

def sigmoid(x):
    return x.sigmoid()

# Example usage with different activation functions for each layer
activation_funcs = [relu, tanh, sigmoid]
my_network = network(num_in=4, layer_sizes=[3, 2, 1], activation_funcs=activation_funcs)

# You can now use my_network for training and prediction with the specified activation functions.
