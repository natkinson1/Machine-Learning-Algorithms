'''Decision Tree Regression Forest Machine Learning Model.'''
import numpy as np

class Leaf:
    '''Leaf Node object to store rows at the end of a decision tree branch'''
    def __init__(self, rows):

        self.prediction = np.mean(rows)

class DecisionNode:
    '''Decision Node object to store question asked and the branchs at the node'''
    def __init__(self, true_branch, false_branch, split_val, column_idx):

        self.true_branch = true_branch
        self.false_branch = false_branch
        self.split_val = split_val
        self.column_idx = column_idx

    def match_condition(self, row):
        '''Compares a row value to the best split value calculated for the
        decision node object.S'''
        if row[self.column_idx] > self.split_val:
            return True

        return False

class DecisionTree:
    '''Creates a Decision Tree object

    Parameters :
    ----------

    max_depth : The maxiumum depth you want the decision tree to go.

    Returns
    -------

    NoneType'''

    def __init__(self, max_depth=1):

        self.max_depth = max_depth
        self.tree = None
        self.predictions = None

    def mse(self, target):
        '''Calculates the mean squared error of a sample compared with its
        sample mean

        Parameters :
        ----------

        target : Array of values

        Returns :
        -------

        Mean Squared Error of target.'''

        error = (target - target.mean()) ** 2

        return error.sum() / len(error)

    def best_split(self, X, y):
        '''Finds the value in a column which minimises the mean squared error

        Parameters :
        ----------

        X : Data Matrix
        y : Target Vector

        Returns :
        -------

        best_true_index : Bool array of all rows which meet best split condition.
        best_false_index : Bool array opposite to best_true_index.
        best_value : Value which minimises mean squared error the most.
        best_column : column index which is minimises the mean squared error the most.'''

        best_error = np.inf
        best_value = None
        best_column = None

        for i in range(X.shape[1]):
            for value in np.unique(X[:, i]):

                true_index = X[:, i] > value
                false_index = X[:, i] <= value

                prob = np.sum(true_index) / len(true_index)

                error = prob * self.mse(y[true_index])\
                        + (1 - prob) * self.mse(y[false_index])

                if error < best_error:
                    best_error = error
                    best_true_index = true_index
                    best_false_index = false_index
                    best_column = i
                    best_value = value

        return best_true_index, best_false_index, best_value, best_column

    def fit(self, X, y, max_depth=None):
        '''Fits the Decision Tree to the dataset.
        Updates the self.tree variable.

        Parameters :
        ----------

        X : Data Matrix
        y : Target vector

        Returns :
        -------

        The Decision Tree
        '''
        if max_depth is None:
            max_depth = self.max_depth


        #Stopping criteria
        if max_depth == 0 or (y == y[0]).all():
            return Leaf(y)

        true_index, false_index, value, col = self.best_split(X, y)


        true_branch = self.fit(X[true_index],
                               y[true_index],
                               max_depth=max_depth - 1)

        false_branch = self.fit(X[false_index],
                                y[false_index],
                                max_depth=max_depth - 1)

        self.tree = DecisionNode(true_branch, false_branch, value, col)

        return DecisionNode(true_branch, false_branch, value, col)

    def _traverse_tree(self, row, node=None):
        '''Moves done the already made tree recursively to see what Leaf Node a row in the
        Data Matrix goes to. Considers only 1 row in the data at a time.
        Is only called in the predict method.

        Parameters :
        ----------
        row : A single row in the data matrix.
        node : (Ignore)

        Returns :
        -------

        node.predict() : The mean of all target values at the Leaf Node'''

        if node is None:
            node = self.tree

        if isinstance(node, Leaf):

            #When at a leaf node
            return node.prediction

        #When at a decision node
        if node.match_condition(row):
            return self._traverse_tree(row, node.true_branch)

        return self._traverse_tree(row, node.false_branch)

    def predict(self, X):
        '''Predicts each row in the datamatrix by calling the _traverse_tree
        method on each row. Appends each prediction to self.predictions.

        Parameters :
        ----------

        X : Data Matrix

        Returns :
        -------

        None'''
        self.predictions = []

        for row in X:
            self.predictions.append(self._traverse_tree(row))

        return np.array(self.predictions)

class RegressionForest:
    '''Ensemble classifier that builds weak regression decision tree estimators.
    Computes the mean of each weak estimator prediction.'''

    def __init__(self,
                 n_estimators=100,
                 random_state=42,
                 max_depth=1,
                 n_features=None,
                 n_samples=None,
                 bootstrap=False):

        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.trees = [self.build_tree() for i in range(self.n_estimators)]
        self.n_features = n_features
        self.n_samples = n_samples
        self.bootstrap = bootstrap
        self.predictions = None

        np.random.seed(self.random_state)

    def build_tree(self):
        '''Builds a decision tree object with max_depth = self.max_depth.
        Function called for each decision tree estimator when RegressionForest
        object to created.'''
        return DecisionTree(max_depth=self.max_depth)

    def fit(self, X, y):
        '''Fits each Decision Tree estimator to the data matrix.

        Parameters :
        ----------

        X : Data Matrix
        y : Target Vector

        Returns :
        -------

        None'''
        for tree in self.trees:

            #choose random set of features
            if self.n_features is None:
                feature_set = tuple(np.arange(X.shape[1]))
            else:
                feature_set = tuple(np.random.choice(np.arange(X.shape[1]),
                                                     self.n_features,
                                                     replace=False))
            #choose random sample set
            if self.n_samples is None:
                sample_set = np.arange(X.shape[0])
            else:
                sample_set = np.random.choice(np.arange(X.shape[0]),
                                              self.n_samples,
                                              replace=self.bootstrap)

            #Fit each tree to the data.
            tree.fit(X[:, feature_set][sample_set], y[sample_set])

    def predict(self, X):
        '''Makes a prediction for each row in the Data Matrix. Takes the mean
        of all prediction made by each decision tree estimator. This method
        updates the self.predictions variable.

        Parameters :
        ----------

        X : Data Matrix

        Returns :
        -------

        self.predictions'''

        self.predictions = np.array([tree.predict(X) for tree in self.trees])

        self.predictions = self.predictions.mean(axis=0)

        # self.predictions = np.zeros(X.shape[0])
        #
        # for tree in self.trees:
        #
        #     self.predictions += tree.predict(X)
        #
        # self.predictions /= self.n_estimators

        return self.predictions
