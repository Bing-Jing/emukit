
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from scipy.interpolate import interp1d

class funcEnv():
    def __init__(self):
        self.curFun = None
        self.maxVal = 0
        self.minVal = float('inf')
    def reset(self,sample_point = 2000,upper_bound = 1, lower_bound = 0):
        X = np.linspace(lower_bound, upper_bound, num=sample_point)[:, None]
        # 2. Specify the GP kernel (the smoothness of functions)
        # Smaller lengthscale => less smoothness
        kernel_var = 1.0
        kernel_lengthscale = 0.05 ## modify to 0.1~0.5
        kernel = kernel_var * RBF(kernel_lengthscale)
        # 3. Sample true function values for all inputs in X
        trueF = self.sample_true_u_functions(X, kernel)
        Y = trueF[0]
        self.curFun = interp1d(X.reshape(-1), Y, kind='cubic')
        self.maxVal = max(self.curFun(X))
        self.minVal = min(self.curFun(X))
        return self.curFun
    def getCurFun(self):
        return self.curFun
    # functions sampled from GP
    def sample_true_u_functions(self,X, kernel):
        u_task = np.empty(X.shape[0])
        mu = np.zeros(len(X)) # vector of the means
        C = kernel.__call__(X,X) # covariance matrix
        u_task[:,None] = np.random.multivariate_normal(mu,C).reshape(len(X),1)
        
        return [u_task]
