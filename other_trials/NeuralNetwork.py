"""
Dunno wtf
"""



import numpy as np

class Dense:
    def __init__(self, units, n_inputs, activation):
        self.weights = np.random.randn(units, n_inputs)
        self.biases = np.zeros((1, units))
        self.activation = activation
    
    def activation(self, output):
        if self.activation == 'relu':
            return np.max((0, output), axis=1, keepdims=True)
        elif self.activation == 'softmax':
            return np.exp(output) / np.sum(np.exp(output), axis=1)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-output))
        elif self.activation == 'linear':
            return output
        
    def forward(self, X):
        output = np.dot(self.weights, X) + self.biases
        self.output = self.activation(output)


class Neural_Network:
    def __init__(self, input):
        self.input = input
    
    def forward_prop(self, layers):
        for i in range(len(layers)):
            layers[i].forward(X)
            X = layers[i].output

    def back_prop(self):
        pass

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: # if sparse
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # if one-hot encoded
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
class Accuracy:
    def calculate(self, output, class_target):
        predictions = np.argmax(output, axis = 1)
        accuracy = np.mean(predictions == class_target)
        print("Accuracy: ", accuracy)

