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

class KernelLogisticRegression(LinearModel):
    
    def __init__(self, kernel: function, lam: float, gamma: int):
        super().__init__()
        self.a = None
        self.lam = lam
        self.gamma = gamma
        self.kernel = kernel
        self.X_t = None
        self.prevK = None
    
    def score(self, X: torch.Tensor, recompute_kernel: bool):
        if recompute_kernel: 
            K = self.kernel(X, self.X_t, self.gamma)
            self.prevK = K
        else:
            K = self.prevK
            
        return K@self.a
    
  
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, m_epochs: int=10, lr: float=0.1):
        m = X.size(0) if isinstance(X, torch.Tensor) else len(X)
        
        # compute the kernel matrix
        K = self.kernel(X, self.X_t, self.gamma)  
        
        # perform the fit
        self.a = torch.zeros(K)
        
        # save the training data: we'll need it for prediction
        self.X_t = X
        
        opt = GradientDescentOptimizer(self)

        # for epoch in range(m_epochs):
        #     s = K @ self.a  # shape: (m,)
        #     loss = -torch.mean(y * torch.log(torch.sigmoid(s) + 1e-8) +
        #                        (1 - y) * torch.log(1 - torch.sigmoid(s) + 1e-8))
        #     loss += self.lam * torch.norm(self.a, p=1)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        
        
    # def predict(self, X):
    #     """
    #     implements eq. 12.3
    #     """
    #     # compute the kernel matrix of the new data with the training data
    #     k = self.kernel(X, self.X_train, **self.kwargs)

    #     # compute the predictions
    #     s = k@self.a
        
    #     return s  
    
    # def loss(self, K: torch.Tensor, y: torch.Tensor):
    #     s = K@self.a
    #     sig = lambda s: 1/(1 + torch.exp(-s))
    #     p1 = y * torch.log(sig(s))
    #     p2 = (1 - y) * torch.log(1 - sig(s))
    #     reg = 1
    #     return -torch.mean(p1 + p2 + reg)

class GradientDescentOptimizer():

    def __init__(self, model: KernelLogisticRegression):
        self.model = model 
        self.prev_w = None
    
    def step(self, X: torch.Tensor, y: torch.Tensor, alpha: float, beta: float):
        """
        Compute one step of the logistic update using the feature matrix X 
        and target vector y. 
        
        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

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