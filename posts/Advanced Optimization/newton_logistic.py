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
            number of features.
            
            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1} 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s =  self.score(X)
        return (s >= 0)*1.0

class LogisticRegression(LinearModel):
    
    def loss(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute the average binary cross-entropy loss for logistic regression.

        ARGUMENTS:
            X, torch.Tensor: Feature matrix of shape (n_samples, n_features).
            y, torch.Tensor: Target labels of shape (n_samples,), values in {0, 1}.

        RETURNS:
            torch.Tensor: Scalar tensor representing the mean binary cross-entropy loss.
        """
        s = self.score(X)
        sig = lambda s: 1/(1 + torch.exp(-s))
        return torch.mean(-y*torch.log(sig(s)) - (1 - y)*torch.log(1 - sig(s)))

    def grad(self, X: torch.Tensor, y: torch.Tensor, alpha: float = 0.002, mini_batch: bool = False):
        """
        Compute the gradient of the logistic regression loss with respect to model weights.

        ARGUMENTS:
            X, torch.Tensor: Feature matrix of shape (n_samples, n_features).
            y, torch.Tensor: Target labels of shape (n_samples,), values in {0, 1}.
            alpha, float: Learning rate (unused in computation, kept for compatibility).
            mini_batch, bool: If True, use a random mini-batch of size 5.

        RETURNS:
            torch.Tensor: Gradient vector of shape (n_features,).
        """
        if mini_batch: 
            k = 5 
            ix = torch.randperm(X.size(0))[:k]
            X = X[ix,:]
            y = y[ix] 
        
        s = self.score(X)
        sig = lambda s: 1/(1 + torch.exp(-s))
        return X.T @ (sig(s) - y) / X.size(0)
    
    def hessian(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute the Hessian matrix (second derivative) of the logistic regression loss.

        ARGUMENTS:
            X, torch.Tensor: Feature matrix of shape (n_samples, n_features).
            y, torch.Tensor: Target labels of shape (n_samples,), values in {0, 1}.

        RETURNS:
            torch.Tensor: Hessian matrix of shape (n_features, n_features).
        """
        s = self.score(X)
        sig = lambda s: 1/(1 + torch.exp(-s))
        diag = sig(s) * (1 - sig(s))
        diag = diag.unsqueeze(1)
        return X.T @ (diag * X) / X.size(0)

class GradientDescentOptimizer():
    def __init__(self, model: LogisticRegression):
        """
        Initialize the optimizer with a logistic regression model.

        ARGUMENTS:
            model, LogisticRegression: The model to optimize.
        """
        self.model = model 
        self.prev_w = None
    
    def step(self, X: torch.Tensor, y: torch.Tensor, alpha: float, beta: float, mini_batch: bool = False):
        """
        Perform one step of (momentum) gradient descent on the model weights.

        ARGUMENTS:
            X, torch.Tensor: Feature matrix of shape (n_samples, n_features).
            y, torch.Tensor: Target labels of shape (n_samples,).
            alpha, float: Learning rate.
            beta, float: Momentum parameter.
            mini_batch, bool: If True, use a random mini-batch.

        RETURNS:
            None
        """
        loss = self.model.loss(X, y)
        if self.prev_w is None:
            self.prev_w = self.model.w.clone()
        
        grad = self.model.grad(X, y, mini_batch=mini_batch)
        new_w = self.model.w - alpha * grad + beta * (self.model.w - self.prev_w)
        self.prev_w = self.model.w.clone()
        self.model.w = new_w
        

class NewtonOptimizer():
    def __init__(self, model: LogisticRegression):
        """
        Initialize the optimizer with a logistic regression model.

        ARGUMENTS:
            model, LogisticRegression: The model to optimize.
        """
        self.model = model 
       
    # def step(self, X: torch.Tensor, y: torch.Tensor, alpha: float):
    #     """
    #     Perform one step of Newton's method for logistic regression.

    #     ARGUMENTS:
    #         X, torch.Tensor: Feature matrix of shape (n_samples, n_features).
    #         y, torch.Tensor: Target labels of shape (n_samples,).
    #         alpha, float: Step size multiplier.

    #     """
    #     grad = self.model.grad(X, y)
    #     hess = self.model.hessian(X, y)
    #     inv_hess = torch.linalg.inv(hess)
    #     self.model.w = self.model.w - alpha * inv_hess @ grad
    
    def step(self, X: torch.Tensor, y: torch.Tensor, alpha: float):
        grad = self.model.grad(X, y)
        hess = self.model.hessian(X, y)
        
        # Regularize Hessian for numerical stability
        damping = 1e-3
        identity = torch.eye(hess.size(0), dtype=hess.dtype, device=hess.device)
        delta = torch.linalg.solve(hess + damping * identity, grad)
        self.model.w = self.model.w - alpha * delta
    
    

