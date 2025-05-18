import torch
from newton_logistic import LogisticRegression

class AdamOptimizer():
    
    def __init__(self, model: LogisticRegression):
        """
        Initialize the Adam optimizer with a logistic regression model.

        ARGUMENTS:
            model, LogisticRegression: The model to optimize.
        """
        self.model = model
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Timestep

    def optim(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, w_0: torch.Tensor = None,
             alpha: float = 0.002, beta_1: float = 0.9, beta_2: float = 0.999):
        """
        Perform Adam optimization on the model weights.

        ARGUMENTS:
            X, torch.Tensor: Feature matrix of shape (n_samples, n_features).
            y, torch.Tensor: Target labels of shape (n_samples,).
            batch_size, int: Number of samples per mini-batch.
            w_0, torch.Tensor or None: Optional initial weights of shape (n_features,).
            alpha, float: Learning rate.
            beta_1, float: Exponential decay rate for the first moment estimates.
            beta_2, float: Exponential decay rate for the second moment estimates.

        RETURNS:
            torch.Tensor: Optimized weights of shape (n_features,).
        """
        # Initialize parameters
        iterations = 1000
        epsilon = 1e-8
        
        n, p = X.shape
        
        if w_0 is None:
            self.model.w = torch.rand(p, dtype=X.dtype)
        else:
            self.model.w = w_0
        
        if self.m is None:
            self.m = torch.zeros(p)
            self.v = torch.zeros(p)
        
        # Perform Adam optimization
        for i in range(iterations):
            self.t += 1
            
            ix = torch.randperm(n)[:batch_size]
            x_i, y_i = X[ix,:], y[ix]
            
            g_t = self.model.grad(x_i, y_i) # get gradients at timestep t
            
            self.m = beta_1 * self.m + (1 - beta_1) * g_t # update biased first moment estimate
            self.v = beta_2 * self.v + (1 - beta_2) * g_t**2 # update biased second moment estimate
            
            m_hat = self.m / (1 - beta_1** self.t) # compute bias-corrected first moment estimate
            v_hat = self.v / (1 - beta_2**self.t) # compute bias-corrected second moment estimate
            
            with torch.no_grad():
                self.model.w -= alpha * m_hat / (torch.sqrt(v_hat) + epsilon)
           
        return self.model.w