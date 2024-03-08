# -*- coding: utf-8 -*-
"""linear_regression
# this is logistic regression implementation in numpy and pytorch
# for simplicity, this is only single variable regression
"""

import torch
import torch.nn as nn
import numpy as np
import random

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression(object):
    def __init__(self, implementation='numpy'):
        self.implementation= implementation
    def fit(self, x ,y):
        if self.implementation == 'numpy':
            self._fit_by_numpy(x,y)
        elif self.implementation == 'pytorch':
            self._fit_by_pytorch(x,y)

    def _fit_by_pytorch(self, x, y):
        #X_test = torch.from_numpy(x.astype(np.float32))
        #y_test = torch.from_numpy(y.reshape(y.shape[0],1).astype(np.float32))
        self.model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid() )

        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        # Train the network
        for epoch in range(100):
            # Forward pass
            y_pred = self.model(torch.from_numpy(X_test.astype(np.float32)))

            # Calculate the loss
            loss = criterion(y_pred.squeeze(), torch.from_numpy(y_test.astype(np.float32)))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # weight and bias (w,b)
        for name, param in self.model.named_parameters():
            print("fit_by_pytorch, weight and bias: ",name, param)


    def _fit_by_numpy(self, x, y):
        #idx = random.randint(0, 149)
        self.w=11
        self.b=1
        self.lr=0.1
        for i in range(100):
            idx = random.randint(0, y.shape[0]-1)
            xi= x[idx]
            yi= y[idx]
            pred= sigmoid(self.w*xi+self.b) # this is to add sigmoid function
            self.w= self.w-(pred-yi)*xi*self.lr
            self.b= self.b-(pred-yi)*1*self.lr
        print("fit_by_numpy, weight and bias: ",self.w,self.b)

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
        prediction= sigmoid(self.w*x+self.b)
        return prediction



# sample use
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    np.random.seed(1234)
    X_test= np.random.randint(9, size=(150, 1))
    y_test= np.random.randint(2, size=(150,))

    lr=LogisticRegression(implementation='pytorch')
    lr.fit(X_test,y_test)
    a = np.array([10])
    a= torch.from_numpy(a.astype(np.float32))

    #print(lr.predict(a, 'pytorch'))
    ##
    lr=LogisticRegression(implementation='numpy')
    lr.fit(X_test,y_test)
    print(lr.predict(10, 'numpy'))
