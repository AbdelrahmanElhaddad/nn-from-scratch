def main():
    # Reading and preprocessing some parts of the data.
    data = pd.read_csv('titanic_train.csv')
    y = data.Survived
    data = data.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)
    data = pd.get_dummies(data)
    data = data.sample(frac=1).reset_index(drop=True)
    mean = data['Age'].mean()
    data['Age'].fillna(mean, inplace=True)

    train, test = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]
    train_y, test_y = train.Survived, test.Survived

    train_X = train.drop(['Survived'], axis=1)
    test_X = test.drop(['Survived'], axis=1)
    network = [
        Dense(10, 128),
        Sigmoid(),
        Dense(128, 64),
        Sigmoid(),
        Dense(64, 32),
        Sigmoid(),
        Dense(32, 16),
        Sigmoid(),
        Dense(16, 8),
        Sigmoid(),
        Dense(8, 1),
        Sigmoid()
    ]
    # 
    epochs = 500
    learning_rate = 0.001
    # training
    for e in range(epochs):
        error = 0
        iter = 0
        print(f"\nepoch:{e}")
        training_accuracy = 0
        for i in range(len(train_X)):
            iter += 1
            # forward
            x = train_X.iloc[i]
            output = np.array([x], dtype=np.float64).T
            ln = 0
            for layer in network:
                ln += 1
                output = layer.forward(output)

            y = np.array(train_y.iloc[i])

            # error
            error += log_loss(y, output)

            # backward
            grad = log_loss_prime(y, output)
            if threshold(output, 0.5) == y:
                training_accuracy += 1
            for layer in reversed(network):
                ln -= 1
                grad = layer.backward(grad, learning_rate)
        error = error / len(train_X)
        training_accuracy /= len(train_X)

        print('\n\n%d/%d, error=%f' % (e + 1, epochs, error))
        print('Training Accuracy: ', training_accuracy)
    print("FINISHED TRAINING. STARTING TESTING\n\n\n")

    # testing
    testing_error = 0
    accuracy = 0
    for i in range(len(test_X)):
        x = test_X.iloc[i]
        output = np.array([x]).T
        for layer in network:
            output = layer.forward(output)
        y_true = np.array(test_y.iloc[i])
        if y_true == threshold(output, 0.5): accuracy += 1
        testing_error += log_loss(y_true, output)
    testing_error /= len(test_X)
    print('Testing Accuracy: ', accuracy / len(test_X))

    print('Total testing error: ', testing_error)

def threshold(x, threshold):
    return int(x > threshold)

from nn import *
import numpy as np
import pandas as pd

main()