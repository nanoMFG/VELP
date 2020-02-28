import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as Kernels  
from scipy.optimize import minimize
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as tf_layer

class Kernel_Optimization():
    def __init__(self, dict_mat=None, kernel_type='RBF', CV=5, X=np.array([[1,2],[2,3],[3,4]]) , y=np.array([[1],[2],[3]]),
                All_material = ['K+','P-']):
        self._kernel_type = kernel_type
        self.All_material = All_material
        kernel = getattr(Kernels,kernel_type)
        self.dict_mat = dict_mat
        
        
        if kernel_type =='ExpSineSquared':
            param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [kernel(length_scale=l,periodicity=p)
                         for l in np.logspace(-2, 2, 500)
                         for p in np.logspace(-2, 2, 500)]}
            
        elif kernel_type =='RBF':
            param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [kernel(length_scale=l)
                         for l in np.logspace(-2, 2, 100)]}
            
        self._CV = CV
        self.kr= GridSearchCV(KernelRidge(), iid=True, cv=self._CV, param_grid=param_grid)
        
        self.X , self.y = X, y

        self.kr.fit(self.X, self.y)
        
    def kr_func(self, x):
        return self.kr.predict(x)
    
    
    def constraint(self, x):    
        ''' Create Constraints for physically-consistent solvent decomposition
            sum_cat x_i = 1.0 & sum_an x_i = 1.0 , x_i > 0 for both cation and anaion
        '''
        
        n_cations = 0
        n_anions = 0
        
        for k in self.All_material:
            if k[-1] =='+':
                n_cations += 1
            else:
                n_anions += 1
        n_constraints = len(self.All_material)+ 2
        
        for cnt, m in enumerate(self.All_material):
            if m[:-1] in self.dict_mat.keys():
                n_constraints -= 1
            if x[cnt] <0 or x[cnt] > 1:
                n_constraints += 1
        val_constraints =  np.zeros((n_constraints))
        
        cat_list = []
        an_list = []
        # active (user selected) materials constraints 
        for k, v in self.dict_mat.items():
            if v =='+':
                cat_list.append(k)
            if v =='-':
                an_list.append(k)
                
        cnt = 2
        for i in range(len(self.All_material)):
            if self.All_material[i][:-1] in cat_list:
                val_constraints[0] += x[i]
            elif self.All_material[i][:-1] in an_list:
                val_constraints[1] += x[i]
            else: 
                val_constraints[cnt] += x[i]
                cnt += 1
            if x[i] < 0 or x[i] > 1:
                val_constraints[cnt] += x[i]
                cnt += 1
                
        val_constraints[0] -= 1.0
        val_constraints[1] -= 1.0
                
        return val_constraints

    def minimize_func(self, optimal, sig,i=0):
        if i==0:
            optimal = self.X[np.random.randint(self.X.shape[0])]
        def funct(x):    
            const =  self.constraint(x)
            f = 0 
            for i in range(len(const)):
                f += sig*max(0.0, const[i]**2)
            return self.kr_func(x) + f
        
        res = minimize(funct, optimal, method='nelder-mead', options={'xtol': 1e-16, 'disp': False, 'maxiter': 1000})
        optimal = res.x
        
        return optimal