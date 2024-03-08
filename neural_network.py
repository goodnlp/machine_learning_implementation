# -*- coding: utf-8 -*-
"""neural_network
1.   two layers, first layer has 3 nodes, followed by sigmoid activation, second layer has 1 node
2.  this is only to show the working mechanism of neural network, for simplicity, it does not include bias term currently
3.   loss function is MSE
"""

import torch
import torch.nn as nn
import numpy as np
import random

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class SimpleNeuralNetwork(object):
    def __init__(self, implementation='numpy'):
        self.implementation= implementation
    def fit(self, x ,y):
        if self.implementation == 'numpy':
            self._fit_by_numpy(x,y)
        elif self.implementation == 'pytorch':
            self._fit_by_pytorch(x,y)

    def _fit_by_pytorch(self, x, y):
        self.model = nn.Sequential(nn.Linear(3, 2), nn.Sigmoid(),nn.Linear(2, 1))

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        # Train the network
        for epoch in range(100):
            # Forward pass
            y_pred = self.model(torch.from_numpy(X_test.astype(np.float32)))

            # Calculate the loss
            loss = criterion(y_pred.squeeze(), torch.from_numpy(y_test.astype(np.float32)))
            # print("fit_by_pytorch, loss: ",loss)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # weight and bias (w,b)
        for name, param in self.model.named_parameters():
            print("fit_by_pytorch, weight and bias: ",name, param)


    def _fit_by_numpy(self, x, y):
        x = x.T
        self.weight_L1 = np.random.rand(2, 3)
        self.weight_L2 = np.random.rand(1, 2)
        for i in range(100):
            p = np.dot(self.weight_L1, x)
            z= sigmoid(p)
            y_pred =np.dot(self.weight_L2, z)

            # back propagation
            error = y_pred - y
            error = error.flatten()
            # only select one sample
            idx = np.random.randint(0,error.shape[0])
            error_idx = error[idx]
            print(error_idx**2)
            #
            self.L2_gradient= z[:, idx]* error_idx
            #
            #t1 = arr[:,idx] * derivative_sigmoid(p[:,idx])[0] * weight_L2[0][0] * error_idx ##
            #t2=  arr[:,idx] * derivative_sigmoid(p[:,idx])[1] * weight_L2[0][1] * error_idx
            #L1_gradient= np.vstack((t1, t2))
            #print(L1_gradient)

            self.L1_gradient = np.dot(x[:, idx].reshape(-1,1), derivative_sigmoid(p[:,idx].reshape(1,-1))) * self.weight_L2 # use star product to do element-wise product
            self.L1_gradient = self.L1_gradient.T
            self.L1_gradient= self.L1_gradient * error_idx
            # update weight
            self.weight_L1 = self.weight_L1 - 0.1 * self.L1_gradient
            self.weight_L2 = self.weight_L2 - 0.1 * self.L2_gradient

    def predict(self, x, implementation):
        if implementation == 'numpy':
            return self._predict_by_numpy(x)
        elif implementation == 'pytorch':
            return self._predict_by_pytorch(x)

    def _predict_by_pytorch(self, x):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x)
        return prediction
    def _predict_by_numpy(self, x):
        p = np.dot(self.weight_L1, x)
        z= sigmoid(p)
        y_pred =np.dot(self.weight_L2, z)
        return y_pred



# sample use
if __name__ == '__main__':
    from sklearn.datasets import make_regression
    X_test, y_test = make_regression(n_samples=1500, n_features=3, noise=1, random_state=42)
    # print("ground truth weight is: ", coef)
    #lr= SimpleNeuralNetwork(implementation='pytorch')
    #lr.fit(X_test,y_test)
    #a = np.array([10,1,1])
    #a= torch.from_numpy(a.astype(np.float32))

    #print(lr.predict(a, 'pytorch'))
    ##
    lr=SimpleNeuralNetwork(implementation='numpy')
    lr.fit(X_test,y_test)
    print(lr.predict([10,1,1], 'numpy'))

