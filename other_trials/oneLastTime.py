"""
Potentially useful file.
"""



import numpy as np
import pandas as pd
"""UPDATE:  Have a look at the forward prop steps(predict function)"""
# input = np.array([1, 2, 3, 4])
# input = np.array([input])
# input = np.reshape(input, (4, 1))
# print(input.shape)
# print(input)

# W1 = np.array([
#     [0.4, 1, 0.4, 0.3],
#     [0.5, 0.3, 0.2, 0.89]
# ])
# # W1 = np.reshape(W1, (4, 2))
# print(W1)
# print(W1.shape)

# print(np.matmul(W1, input))
def main():
    # input = np.array([1, 2, 3, 4, 5])
    # input = np.reshape(input, (5, 1))
    data = pd.read_csv('hp_train.csv')
    y = data.SalePrice
    data_features = data.columns
    features = list()
    for i in data_features:
        if data[i].dtype in ['int64', 'float64']:
            if data[i].corr(y) > 0.5:
                # print(f"{i}: {data[i].corr(y)}")
                features.append(i)
    data = data[features]
    train, test = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]
    train_y, test_y = train.SalePrice, test.SalePrice
    features.remove('SalePrice')
    train_X, test_X = train[features], test[features]
    print(train_X.head())
    network = [
        Dense(10, 16),
        Tanh(),
        Dense(16, 8),
        Tanh(),
        Dense(8, 1)
    ]
    # 
    epochs = 100
    learning_rate = 0.01
    print(train_X.shape[1])
    # training
    for e in range(epochs):
        error = 0
        for i in range(train_X.shape[0]):
            # forward
            x = train_X.loc[i]
            output = np.array([x]).T
            # print(output)
            # print("LMAOOOOOOO")
            
            # for layer in network:
            #     output = layer.forward(output)
            output = predict(network, output)
            # print(output)
            y = np.array(train_y.iloc[i])
            # error
            error += mse(y, output)
            # backward
            grad = mse_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= train_X.shape[0]
        print('%d/%d, error=%f' % (e + 1, epochs, error))



    print('I am a random guess')
    print(np.array(train_X.iloc[49]))
    pred = float(predict(network, np.array([train_X.iloc[60]]).T))
    print("output: ", pred, " true: ", train_y.iloc[49])
    print("difference in values = ", train_y.iloc[49] - pred)



class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self):
        pass

class Dense(Layer):
    def __init__(self, n_input, n_output):
        self.weights = np.random.randn(n_output, n_input)
        self.biases = np.random.randn(n_output, 1)

    def forward(self, input):
        self.input = input
        output = np.dot(self.weights, self.input) + self.biases
        # print("dense:\n", output, "\n", output.shape)
        return output
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        # print(self.weights)
        return input_gradient
        

class Activation(Layer):
    def __init__(self, activation, activaiton_prime):
        self.activation = activation
        self.activation_prime = activaiton_prime
    
    def forward(self, input):
        self.input = input
        output = self.activation(self.input)
        # print("activation", output, "\n", output.shape) 
        return output
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Linear(Activation):
    def __init__(self):
        linear = lambda x: x
        linear_prime = lambda x: [1 if x > 0 else 0]
        super().__init__(linear, linear_prime)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    def __init__(self):
        def relu(input):
            for x in input:
                x = max(x, 0)
            return input
        def relu_prime(input):
            for x in input:
                if x > 0:
                    x = 1
                else:
                    x = 0
            return input
        super().__init__(relu, relu_prime)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def predict(network, input):
    output = input
    for layer in network:
        # print(layer)
        output = layer.forward(output)
    return output
main()

