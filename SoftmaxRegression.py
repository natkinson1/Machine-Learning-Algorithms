import numpy as np
import pandas as pd

class SoftmaxRegression():
    
    def __init__(self, iterations=100, learning_rate=0.01, verbose=False, random_state=42, tol=0.0001):
        
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.theta = 0
        self.random_state = random_state
        self.tol = tol
    
    def fit(self, X, y):
        
        if type(X) == pd.core.frame.DataFrame: 
            X = X.values
            
        if type(y) == pd.core.series.Series:
            y = y.values
        
        #Number of classes
        n_classes = len(np.unique(y))
        
        #Number of features 
        n_feat = X.shape[1]
        
        #Initialise matrix of random numbers
        np.random.seed(self.random_state)
        theta = np.random.randn(n_feat, n_classes)
        
        def __softmax__(logits):
            """Estimates the probability an instances belongs to a class.
            
            logits : Scores of instance belonging to a class
            ----------
            returns : (n,k) matrix of estimated probabilities"""
            
            exp = np.exp(logits)
            exp_sum = np.sum(exp, axis=1, keepdims=True) + 1e-7
            
            return exp / exp_sum
        
        def __one_hot_encode__(target):
            """One hot encodes target vector 
            
            target : target vector
            ----------
            returns : (n,k) one hot encoded matrix"""
            
            obs = len(target) #Number of observations
            n_classes = len(list(set(y))) #Number of classes
            
            one_hot = np.zeros((obs, n_classes))
            one_hot[np.arange(obs), target] = 1
            
            return one_hot
            
        
        i=0
        y_target_matrix = __one_hot_encode__(y)
        n = len(X)
        best_loss = np.infty
        
        while i < self.iterations: #Number of iterations
            
            logits = X.dot(theta) #Scores
            #Stops big values being exponented
            #Due to redundency property of theta
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = __softmax__(logits)
            error = probs - y_target_matrix
            loss = -np.mean(np.sum(y_target_matrix * np.log(probs + 1e-7), axis=1))
            
            if self.verbose == True:
                if i % (self.iterations*0.25) == 0:
                    print("Iteration : {}, Log loss : {}".format(i, loss))
            
            gradient = 1/n*X.T.dot(error)
            
            theta = theta - self.learning_rate*gradient
            
            if abs(best_loss - loss) < self.tol:
                print("Iteration : {} Loss : {}".format(i, loss))
                print("Early Stopping")
                break
            
            if loss < best_loss:
                best_loss = loss
            
            i += 1
        
        self.theta = theta
            
    
    def predict(self, X):
        """Predicts the class of an instance based on training data"""
        
        if type(X) == pd.core.frame.DataFrame:
                X = X.values
        
        def __softmax__(logits):
            """Estimates the probability an instances belongs to a class.
            
            logits : Scores of instance belonging to a class
            ----------
            returns : (n,k) matrix of estimated probabilities"""
            
            exp = np.exp(logits)
            exp_sum = np.sum(exp, axis=1, keepdims=True)
            
            return exp / exp_sum
        
        logits = X.dot(self.theta)
        probs = __softmax__(logits)
        y_predicted = np.argmax(probs, axis=1)
        
        return y_predicted