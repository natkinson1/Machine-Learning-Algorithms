class RidgeRegression():
    
    def __init__(self, alpha, gradient_descent=False, iterations=100, learning_rate=0.1):
        
        self.coef = None
        self.gradient_descent = gradient_descent
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.alpha = alpha
        
    def fit(self, X, y):
        """
        Trains a Ridge Regression Model to a dataset.
        
        Parameters:
        -----------
        X : Dataset
        y : Target
        
        Returns: None"""
        
        n_feat = X.shape[1]
        n_obs = X.shape[0]
        
        self.coef = np.random.randn(n_feat, 1)
        
        if self.gradient_descent:
            
            i = 0
            
            while i < self.iterations:
                #Gradient Descent
                gradient = (2/n_obs)*X.T.dot(X.dot(self.coef)-y) + self.alpha*np.sum(self.coef)
                self.coef = self.coef - self.learning_rate*gradient
                
                i += 1
            
        else:
            #Compute analytically
            self.coef = np.linalg.inv(X.T.dot(X) + self.alpha*np.identity(n_feat)).dot(X.T).dot(y)
            
    def predict(self, X):
        """
        Predicts the target based on a dataset.
        
        Parameters:
        -----------
        X : Dataset
        
        Returns: None"""
        
        return X.dot(self.coef)