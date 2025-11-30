"""
Current Problem:
    o Why doesn't Backpropagation work and why doesn't the network learn properly?
    o How could the output of such activation functions match the big price of a house?
    o Why does ReLU ruin the dimentions while Tanh works properly?
"""

"""
Probable Solutions:
    o Research for a clear answer of how could a house price (big number) come out of such a thing of (SOP)ing small numbers. # That could solve everything we're looking for. It could all be just some misunderstading of an important concept.
    o The problem is in the activation functions.
    o The problem is in the backpropagation.
"""



def main():
    data = pd.read_csv('hp_train.csv')
    y = data.SalePrice
    data_features = data.columns
    features = list()
    for i in data_features:
        if data[i].dtype in ['int64', 'float64']:
            if data[i].corr(y) > 0.5:
                features.append(i)
    data = data[features]
    data = data.sample(frac=1).reset_index(drop=True)
    train, test = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]
    train_y, test_y = train.SalePrice, test.SalePrice

    max_price_train = np.max(train_y)
    max_price_test = np.max(test_y)

    train_y = normalize(train_y)
    test_y = normalize(test_y)

    features.remove('SalePrice')
    train_X, test_X = train[features], test[features]
    print(len(train_X.loc[0]))
    #
    network = [
        Dense(10, 8),
        ReLU(),
        Dense(8, 1),
        Tanh()
    ]
    # 
    epochs = 10
    learning_rate = 0.001
    # training
    for e in range(epochs):
        error = 0
        iter = 0
        for i in range(len(train_X)):
            iter += 1
            print(f"epoch:{e}, iter:{iter}")
            # forward
            x = train_X.loc[i]
            output = np.array([x]).T
            ln = 0
            for layer in network:
                ln += 1
                output = layer.forward(output)
            print("Prediction: ", unnormalize(output, max_price_train))
            y = np.array(train_y.iloc[i])
            print("True: ", unnormalize(y, max_price_train))
            # error
            error += mse(y, output, max_price_train)
            print(f"iter:{iter}, error:{error}")
            # backward
            grad = mse_prime(y, output, max_price_train)
            for layer in reversed(network):
                ln -= 1
                grad = layer.backward(grad, learning_rate)

        # error /= 1960
        print('%d/%d, error=%f' % (e + 1, epochs, error))







import numpy as np
import pandas as pd


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
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
    

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
        pass

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

def mse(y_true, y_pred, max_value):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred, max_value):
    return 2 * (y_pred - y_true) / np.size(y_true) 

def normalize(x):
    return np.tanh(x)


def unnormalize(x, max_value):
    return (x + 1) / 2 * max_value


"""finish the unnormalization"""
main()

