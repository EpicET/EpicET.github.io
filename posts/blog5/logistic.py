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
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        return X@self.w

    def predict(self, X: torch.Tensor):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.
            
            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1} 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s =  self.score(X)
        return (s >= 0)*1.0

class LogisticRegression(LinearModel):
    
    def loss(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute the empirical risk L(w), using logistic loss.
        
        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            L(w), torch.Tensor: the empirical risk.
        """
      
        s = self.score(X)
        sig = lambda s: 1/(1 + torch.exp(-s))
        return torch.mean(-y*torch.log(sig(s)) - (1 - y)*torch.log(1 - sig(s)))

    def grad(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute the gradient of empirical risk ∇L(w)
        
        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.
            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        RETURNS:
            ∇L(w), torch.Tensor: the gradient of empirical risk.
        """
        s = self.score(X)
        sig = lambda s: 1/(1 + torch.exp(-s))
        return X.T @ (sig(s) - y) / X.size(0)
    

class GradientDescentOptimizer():

    def __init__(self, model: LogisticRegression):
        self.model = model 
        self.prev_w = None
    
    def step(self, X: torch.Tensor, y: torch.Tensor, alpha: float, beta: float):
        """
        Compute one step of the logistic update using the feature matrix X 
        and target vector y. 
        
        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
            alpha, float: the learning rate.
            beta, float: the momentum term.
        """
        loss = self.model.loss(X, y)
        # print(f"Loss: {loss.item()}")
        if self.prev_w is None:
            self.prev_w = self.model.w.clone()

        grad = self.model.grad(X, y)
        new_w = self.model.w - alpha * grad + beta * (self.model.w - self.prev_w)
        self.prev_w = self.model.w.clone()
        self.model.w = new_w