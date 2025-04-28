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

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s =  self.score(X)
        return (s >= 0)*1.0

class Perceptron(LinearModel):

    def loss(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, 
        where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: You are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} 
        -- otherwise none of the formulas will work right! An easy to to make this conversion is: y_ = 2*y - 1
        """
        y_i = 2*y - 1 # convert to {-1, 1}
        s_i = self.score(X)
        x_i = s_i * y_i > 0
        return 1 - (1.0 * x_i).mean()

    def grad(self, X: torch.Tensor, y: torch.Tensor, alpha: float, mini_batch: bool):
        """
        The calculation to update the weight

        Args:
            X, torch.Tensor: a feature matrix with a single row
            y, torch.Tensor: target vector
            alpha, float: learning rate

        Returns:
            w, torch.Tensor: weight vector
        """

        # Mini-batch Gradient Descent - Updates k points at a time
        if mini_batch:
            s = self.score(X)
            return (alpha / X.size(0)) * (((s * (2*y - 1) < 0).float() * (2*y - 1)).unsqueeze(1) * X).sum(dim=0)
        
        # Original Perceptron Gradient Descent - Updates one point at a time
        x_i = X[0] # get a single row of X
        y_i =  y[0]   
        s_i = self.score(x_i) # get the score 
        return -1*(s_i * (2*y_i-1) < 0) * (2*y_i-1) * x_i
    
class PerceptronOptimizer:

    def __init__(self, model: Perceptron):
        self.model = model 
    
    def step(self, X: torch.Tensor, y: torch.Tensor, alpha: float = 0.0, mini_batch: bool = False):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        loss = self.model.loss(X, y)
        grad = self.model.grad(X, y, alpha, mini_batch)
        if(mini_batch):
            self.model.w += grad
        else:
            self.model.w -= grad