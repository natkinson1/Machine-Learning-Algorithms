import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy.linalg import inv
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize

#With help from the webstite: http://krasserm.github.io/2018/03/19/gaussian-processes/


class GaussianProcessRegression:

    def __init__(self, kernel=None, alpha=1e-3, random_state=42):

        if kernel is None:
            self.kernel = self._rbf_kernel
        else:
            self.kernel = kernel
        self.alpha = alpha
        self.random_state = random_state
        self.training_data = None
        self.label_data = None
        self.mu = None
        self.cov = None
        self.theta = [1, 1]

    def _rbf_kernel(self, x_1, x_2, l=1, sigma=1):
        '''Radial Basis Kernel Function
        Default Kernel for Gaussian Process Regression'''

        val = np.sum(x_1**2, 1).reshape(-1, 1) + \
              np.sum(x_2**2, 1) - 2 * np.dot(x_1, x_2.T)


        return sigma**2 * np.exp(-0.5 / l**2 * val)

    def optimisation_method(self, X, y, theta, noise):
        '''Parameters :
           ----------

           X : Data Matrix
           y : label vector
           theta : parameter set of dict type (define before)'''

        def stable_log_likelihood(theta):

            K = self._rbf_kernel(X, X, l=theta[0], sigma=theta[1]) + noise**2 * np.eye(len(X))
            L = cholesky(K)

            return np.sum(np.log(np.diagonal(L))) + \
                   0.5 * y.T.dot(lstsq(L.T, lstsq(L, y)[0])[0]) + \
                   0.5 * len(X) * np.log(2*np.pi)

        def log_likelihood(theta):

            K = self._rbf_kernel(X, X, l=theta[0], sigma=theta[1]) + \
                noise**2 * np.eye(len(X))

            return 0.5 * np.log(det(K)) + \
                   0.5 * y.T.dot(inv(K).dot(y)) + \
                   0.5 * len(X) * np.log(2*np.pi)

        return log_likelihood

    def _posterior_predictive(self, X_s, X, y, l=1, sigma=1, noise=1e-8):

        C = self._rbf_kernel(X, X, l, sigma) + noise**2 * np.eye(len(X))
        k = self._rbf_kernel(X, X_s, l, sigma)
        c = self._rbf_kernel(X_s, X_s, l, sigma) + 1e-8 * np.eye(len(X_s))
        C_inv = inv(C)

        mu = k.T.dot(C_inv).dot(y)
        cov = c - k.T.dot(C_inv).dot(k)

        return mu, cov

    def fit(self, X, y):

        self.training_data = X
        self.label_data = y


        minimum = minimize(self.optimisation_method(X, y, self.theta, self.alpha),
                           [1, 1],
                           bounds=((1e-5, None), (1e-5, None)),
                           method='L-BFGS-B')

        l_opt, sigma_opt = minimum.x
        self.theta = [l_opt, sigma_opt]



    def predict(self, X):
        self.mu, _ = self._posterior_predictive(X,
                                                self.training_data,
                                                self.label_data,
                                                l=self.theta[0],
                                                sigma=self.theta[1],
                                                noise=self.alpha)

        return self.mu
