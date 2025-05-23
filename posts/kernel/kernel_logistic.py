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
        # print("Weight a shape:", self.a.shape)
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
    def __init__(self, kernel, lam: float, gamma: int):
        self.lam = lam
        self.gamma = gamma
        self.kernel = kernel
        self.X_t = None
        self.prevK = None
        self.a = None
        self.prev_a = None
    
    def score(self, K: torch.Tensor):
        """
        Computes the model's output scores for the given kernel matrix.

        Args:
            K (torch.Tensor): The kernel matrix of shape (n_samples, n_train_samples).

        Returns:
            torch.Tensor: The predicted scores for each sample.
        """
        if self.a is None: 
            self.a = torch.rand(K.shape[1])  # matches number of training samples
        return K @ self.a

    def grad(self, K: torch.Tensor, y: torch.Tensor, m: int):
        """
        Computes the gradient of the regularized logistic loss with respect to the model parameters.

        Args:
            K (torch.Tensor): The kernel matrix of shape (m, n), where m is the number of samples and n is the number of features or basis functions.
            y (torch.Tensor): The target labels tensor of shape (m,), with values typically in {0, 1}.
            m (int): The number of training samples.

        Returns:
            torch.Tensor: The gradient of the loss with respect to the model parameters, of shape (n,).
        """
        s = K @ self.a
        sig = 1 / (1 + torch.exp(-s))
        grad_loss = K @ (sig - y) / m + self.lam * self.a
        return grad_loss
  
    def prediction(self, X: torch.Tensor, recompute_kernel: bool = True):
        """
        Computes the predicted probabilities for input samples using the kernel logistic regression model.

        Args:
            X (torch.Tensor): Input data tensor of shape (n_samples, n_features).
            recompute_kernel (bool): If True, recompute the kernel matrix between X and training data; 
                                     if False, use the previously computed kernel matrix.

        Returns:
            torch.Tensor: Predicted probabilities for each input sample, as a 1D tensor.
        """
        if recompute_kernel: 
            K = self.kernel(X, self.X_t, self.gamma)
            self.prevK = K
        else:
            K = self.prevK
        s = K @ self.a
        probs = 1 / (1 + torch.exp(-s))
        return probs
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, m_epochs, lr, beta: float = 0.0):
        """
        Fits the kernel logistic regression model to the provided training data using gradient descent with optional momentum.
        Args:
            X (torch.Tensor): Input feature tensor of shape (n_samples, n_features).
            y (torch.Tensor): Target labels tensor of shape (n_samples,).
            m_epochs (int): Number of training epochs.
            lr (float): Learning rate for gradient descent.
            beta (float, optional): Momentum coefficient. Default is 0.0.
        """
        m = X.size(0)

        if self.X_t is None:
            self.X_t = X

        K = self.kernel(X, self.X_t, self.gamma)
        
        if self.a is None:
            self.a = torch.zeros(K.shape[1])  # num training points
        if self.prev_a is None:
            self.prev_a = self.a.clone()

        for epoch in range(m_epochs):
            grad_a = self.grad(K, y, m)
            new_a = self.a - lr * grad_a + beta * (self.a - self.prev_a)
            self.prev_a = self.a.clone()
            self.a = new_a
