import numpy as np
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# built linear regression model
class LinearRegression(object):
    def __init__(self):
        self.input = None
        self.output = None
        self.n_features = None
        self.n_targets = None
        
        self.coef_ = None
    
    def cos_function(self, predictions, lables):
        n_samples = predictions.shape[0]
        return 0.5 * (np.square(lables - predictions).sum() + self.lambd * np.square(self.coef_).sum()) / n_samples
      
    # training linear regression model
    def fit(self, X, Y, max_iters=1000000, lr=0.01, verbose=False, print_freq=20, tol=1e-6, lambd=1e-2):
        self.input = copy.deepcopy(X)
        self.output = copy.deepcopy(Y)
        self.lambd = lambd
        self.learningrate = lr
        self.epochs = max_iters
        self.tol = tol
        n_samples = X.shape[0]
        
        X = np.matrix(X)
        Y = np.matrix(Y.reshape(n_samples, -1))
        
        self.n_features = X.shape[1]
        self.n_targets = Y.shape[1]
        
        self.coef_ = np.mat(np.zeros([self.n_features,1]))
        self.intercept_ = 0.
        
        lr0 = copy.deepcopy(lr)
        decay_rate = 0.9
        decay_period = 2
        
        pre_loss = float('inf')
        for iter in range(1, max_iters+1):
            # learning rate decay 
            if iter > decay_period:
                lr = lr0 / (1 + decay_rate * (iter - decay_period))
            Y_pred = self.predict(X)

            self.coef_ = self.coef_ - lr * (np.dot(X.T, (Y_pred - Y)) + lambd * self.coef_) / n_samples
            self.intercept_ = self.intercept_ - lr * (Y_pred-Y).mean()
            loss = self.cos_function(Y_pred, Y)
            
            if verbose and (iter % print_freq == 0 or iter == max_iters):
                print('iteration:%d\t loss:%lf' %(iter, loss))
                if pre_loss - loss < tol:
                    break
                pre_loss = loss
            
        
    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    # 'R2 score'
    def R2_score(self, X, Y):
        X = np.matrix(X)
        Y = np.matrix(Y.reshape(-1, self.n_targets))
        Y_pred = self.predict(X)
        Y_pred = Y_pred.reshape(-1)
        Y = Y.reshape(-1)
        SSR = np.sum(np.square(Y_pred - np.mean(Y)))
        SST = np.sum(np.square(Y - np.mean(Y)))
        return (SSR/SST)
    
    # 'MSE'
    def MSE(self, X, Y):
        X = np.matrix(X)
        Y = np.matrix(Y.reshape(-1, self.n_targets))
        Y_pred = self.predict(X)
        return np.square(Y - Y_pred).mean()
    