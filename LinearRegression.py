class LinearRegression():
    
    def __init__(self, gradient_descent=False, iterations=100, learning_rate=0.01):
        
        self.coef = None
        #For large datasets
        self.gradient_descent = gradient_descent
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        """Fits a Linear Regression model to the data
        
        X : Data Matrix
        y : target
        ----------------
        return : None"""
        
        #Number of features in data
        n_feat = X.shape[1]
        n_obs = X.shape[0]
        #Set random Seed
        np.random.seed(42)
        
        #Resize shape of target vector
        y = np.resize(y, (len(y), 1))
        
        if self.gradient_descent:
            
            self.coef = np.random.randn(n_feat, 1)
            i = 0
            #Batch Gradient Descent
            while i < self.iterations:
                
                gradient = (2/n_obs)*X.T.dot(X.dot(self.coef)-y)
                self.coef = self.coef - self.learning_rate*gradient
                i += 1
                
        else:
            #Compute coefficents analytically
            self.coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
    def predict(self, X):
        """Predicts the target for a data matrix
        
        Parameters:
        -----------
        
        X : Data Matrix
        
        Return:
        -------
        y_pred : Array of predicted values
        """
        
        return X.dot(self.coef)