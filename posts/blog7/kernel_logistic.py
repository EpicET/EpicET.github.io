import torch



class LinearModel:

    def __init__(self):
        self.a = None # used to be self.w

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
        if self.a is None: 
            self.a = torch.rand((X.size()[1]))
        # print("Weight a shape:", self.a.shape)
        return X@self.a

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
    def __init__(self, kernel, lam: float, gamma: int):
        self.lam = lam
        self.gamma = gamma
        self.kernel = kernel
        self.X_t = None
        self.prevK = None
    
    # Compute loss L(a) 
    def loss(self, K: torch.Tensor, y: torch.Tensor):
        s = self.score(K)
        sig = lambda s: 1/(1 + torch.exp(-s))
        loss = -y*torch.log(sig(s)) - (1 - y)*torch.log(1 - sig(s))
        reg = self.lam * torch.norm(self.a, p=1)
        return torch.mean(loss + reg) 
    
    def grad(self, K: torch.Tensor, y: torch.Tensor):
        s = self.score(K)
        sig = lambda s: 1/(1 + torch.exp(-s))
        return K @ (sig(s) - y) / K.size(0)
     
    def prediction(self, X: torch.Tensor, recompute_kernel: bool):
        if recompute_kernel: 
            K = self.kernel(X, self.X_t, self.gamma)
            self.prevK = K
        else:
            K = self.prevK
        
        return self.score(K)
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, m_epochs, lr):
        m = X.size(0) if isinstance(X, torch.Tensor) else len(X) 
       # print("X shape:", X.shape)
        
        if self.X_t is None:
            self.X_t = X
        
        # compute the kernel matrix
        K = self.kernel(X, self.X_t, self.gamma)  
        # print("K shape:", K.shape)
        
        # perform the fit
        self.a = torch.zeros(m)
        # print("Weight a shape:", self.a.shape)
        
        # save the training data: we'll need it for prediction
        self.X_t = X
        
        # use our own optimizer?
        opt = GradientDescentOptimizer(model=self)

        for epoch in range(m_epochs):
            loss = self.loss(K, y)
            
            if epoch % 100000 == 0:  # adjust this number as needed
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
            opt.step(K, y, alpha=lr, beta=0.9) # What is beta supposed to be?
   
         
class GradientDescentOptimizer():

    def __init__(self, model: KernelLogisticRegression):
        self.model = model 
        self.prev_a = None # used be self.prev_w
    
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
        if self.prev_a is None:
            self.prev_a = self.model.a.clone()
        loss = self.model.loss(X, y)
        grad = self.model.grad(X, y)
        new_a = self.model.a - alpha * grad + beta * (self.model.a - self.prev_a)
        self.prev_a = self.model.a.clone()
        self.model.a = new_a
        
