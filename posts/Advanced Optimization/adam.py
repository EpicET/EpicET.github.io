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
        Compute the empirical risk L(w), using logistic loss.
        
        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.  

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
            number of features. 
            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        RETURNS:
            ∇L(w), torch.Tensor: the gradient of empirical risk.
        """
        s = self.score(X)
        sig = lambda s: 1/(1 + torch.exp(-s))
        return X.T @ (sig(s) - y) / X.size(0)
    
    def hessian(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute the hessian of empirical risk ∇²L(w)
        
        ARGUMENTS
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. 
            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        RETURNS:
            ∇²L(w), torch.Tensor: the hessian of empirical risk.
        """
        
        s = self.score(X)
        sig = lambda s: 1/(1 + torch.exp(-s))
        diag = sig(s) * (1 - sig(s))
        diag = diag.unsqueeze(1)
        return X.T @ (diag * X) / X.size(0)


class AdamOptimizer():
    
    def __init__(self, model: LogisticRegression):
        self.model = model
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Timestep
    
    def step(self, params: list[torch.Tensor], batch_size: int, w_0: torch.Tensor = None, 
             alpha: float = 0.002, beta_1: float = 0.9, beta_2: float = 0.999):
        """
        Perform one step of the Adam optimization algorithm.

        ARGUMENTS:
            params, list[Tensor]: params[0] = X and params[1] = y
            batch_size, int: the batch size for computing gradients. Used for stochastic gradient descent.
            w_0, torch.Tensor: The initial weight vector. Defaults to None and is randomly initialized with correct dimensions
            alpha, float (optional): the step size. Defaults to 0.002.
            beta_1, float (optional): Exponential decay rate. Defaults to 0.9.
            beta_2, float (optional): Exponential decay rate. Defaults to 0.999.
        """
        # Initialize parameters
        iterations = 1000
        epsilon = 1e-8
        
        X, y = params
        n, p = X.shape
        
        ix = torch.randperm(n)[:batch_size]
        x_i, y_i = X[ix,:], y[ix]
        
        if w_0 is None:
            self.model.w = torch.rand(p, dtype=X.dtype)
        else:
            self.model.w = w_0
        
        if self.m is None:
            self.m = torch.zeros(p)
            self.v = torch.zeros(p)
        
        ix = torch.randperm(n)[:batch_size]
        x_i, y_i = X[ix,:], y[ix]
        
        # Perform Adam optimization
        for i in range(iterations):
            t += 1
            g_t = self.model.grad(x_i, y_i) # get gradients at timestep t
            
            self.m = beta_1 * self.m + (1 - beta_1) * g_t # update biased first moment estimate
            self.v = beta_2 * self.v + (1 - beta_2) * g_t**2 # update biased second moment estimate
            
            m_hat = self.m / (1 - beta_1**t) # compute bias-corrected first moment estimate
            v_hat = self.v / (1 - beta_2**t) # compute bias-corrected second moment estimate
            
            self.model.w -= alpha * m_hat / (torch.sqrt(v_hat) + epsilon)
           
        return self.model.w

