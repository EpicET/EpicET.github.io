import torch
class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X: torch.Tensor):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        
        return X@self.w

class MyLinearRegression(LinearModel):
    
    def predict(self, X: torch.Tensor):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s =  self.score(X)
        return s
    
    def loss(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute the mean squared error. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. 

            y, torch.Tensor: the target vector.  y.size() = (n,).
        """
        s = self.score(X)
        return ((s - y)**2).mean()
    
class OverParameterizedLinearRegressionOptimizer():
    def __init__(self, model: MyLinearRegression):
        self.model = model

    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the model to the data. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the target vector.  y.size() = (n,). 
        """
        
        self.model.w = torch.linalg.pinv(X) @ y
        