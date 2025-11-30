"""
TODO:
    1. Implement the softmax function and it's derivative.
    2. Implement the cost function.
    3. Find some dataset and bang your head against the wall for a couple of days :>
    4. Once finished, find out wtf is wrong with having different activation functions in our network.
"""


import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        # TODO: return output
        pass
    
    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.array(np.random.randn(output_size, input_size), dtype=np.float64)
        self.bias = np.array(np.random.randn(output_size, 1), dtype=np.float64)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= np.array(learning_rate * weights_gradient, dtype=np.float64)
        self.bias -= np.array(learning_rate * output_gradient, dtype=np.float64)
        return input_gradient
    

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = np.array(input, dtype=np.float64)
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(x, 0)
        relu_prime = lambda x: np.where(x > 0, 1, 0)
        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)
        
    def sigmoid(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def sigmoid_prime(self, x):
        return self.output * (1 - self.output)
##################
class Softmax(Activation):
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)

    def softmax(self, z):
        # Shift the input `z` by subtracting the max value for numerical stability
        shift_z = z - np.max(z, axis=1, keepdims=True)
        # Calculate the exponentials
        exp_z = np.exp(shift_z)
        # Sum of exponentials for each row (each sample in the batch)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        # Softmax result for each sample in the batch
        return exp_z / sum_exp_z
    
    def softmax_prime(self, softmax_output):
        # Reshape softmax_output to be a column vector
        s = softmax_output.reshape(-1, 1)  # Column vector
        # Create the Jacobian matrix
        jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
        return jacobian_matrix
    
    
##################

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def log_loss(y_true, y_pred):
    safe_input = np.maximum(y_pred, 1e-7)
    return -(y_true * np.log(safe_input) + (1 - y_true) * np.log(safe_input))

def log_loss_prime(y_true, y_pred):
    safe_input = np.maximum(y_pred, 1e-7)
    output = -(y_true / (safe_input)) + ((1 - y_true) / (1 - safe_input))
    return output

def normalize(x):
    return np.tanh(x)

def unnormalize(x, max_value):
    return (x + 1) / 2 * max_value

