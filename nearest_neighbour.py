import numpy as np

class NearestNeighbourRegression:
    
    def __init__(self, n_neighbors, distance_metric='minkowski', p=2):
        
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.p = p
        self.X = None
        self.y = None
        self.predictions = None
        
    def _distance_metric(self, row1, row2):
        
        def euclidean(row1, row2):
            
            return np.sqrt((row1 - row2) ** 2)
        
        def cosine_distance(row1, row2):
            
            return (row1 * row2).sum() / np.sqrt((row1**2).sum()) * np.sqrt((row2**2).sum())
        
        def minkowski_distance(row1, row2):
            
            return (((abs(row1 - row2)) ** self.p).sum()) ** (1 / self.p)
        
        if self.distance_metric == 'euclidean': return euclidean(row1, row2)
        elif self.distance_metric == 'cosine': return cosine_distance(row1, row2)
        elif self.distance_metric == 'minkowski': return minkowski_distance(row1, row2)
        else: return '{} is an unknown distance metric,\
                     please use euclidean or cosine'.format(self.distance_metric)
        
    def _get_neighbours(self, test_row):
        
        distances = list()
        
        for i, row in enumerate(self.X):
            
            dist = self._distance_metric(row, test_row)
            
            distances.append((i, dist))
        
        distances = sorted(distances, key = lambda x: x[1])
        
        return [i[0] for i in distances[:self.n_neighbors]]
        
        
    def fit(self, X, y):
        
        self.X = X
        self.y = y

    
    def predict(self, X):
        
        self.predictions = np.array([])
        
        for row in X:
            
            idx = self._get_neighbours(row)
            pred = np.mean(self.y[idx])
            self.predictions = np.append(self.predictions, pred)
            
        return self.predictions