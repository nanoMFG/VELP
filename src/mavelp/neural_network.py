import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as Kernels  
from scipy.optimize import minimize
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as tf_layer

class DNN():
    def __init__(self, dict_mat=None, dnn_features=[20,10, 0.75, 0.001, 1000000], activation_name=['tanh', 'tanh'], X=None, y=None, All_material = ['K+','P-']):
        
        self.All_material = All_material
        tf.reset_default_graph()
        self.dict_mat = dict_mat
        self.activation_name = activation_name
        
        # network width
        self.nodes_hidden = dnn_features[:2]
        
        # keep_prob, learning rate
        self.keep_prob = dnn_features[2]
        self.lr = dnn_features[3]
        
        # num of training step and feature dimension
        self.num_step = dnn_features[4]
        self.n_dims = len(self.All_material)
        
        # input and output placeholders
        self.x_placeholder = tf.placeholder(tf.float32, [None, self.n_dims])
        self.y_placeholder = tf.placeholder(tf.float32, [None, 1])
        
        # keep_prob, learning rate placeholer
        self.kp_placeholder = tf.placeholder(tf.float32,[])
        self.lr_placeholder = tf.placeholder(tf.float32,[])
        
        # predicted value 
        self.predictions = self.mlp(self.x_placeholder, self.kp_placeholder)
        
        # loss function
        self.loss = self.loss_dnn(self.predictions, self.y_placeholder)
        
        # read data
        self.X , self.y = X, y
        
        # minimization 
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder).minimize(self.loss)
        
        # Create Session and Initialize Variables
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
    def kr_func(self, x):
        x = x.reshape((-1, len(self.All_material)))
        return self.session.run(self.predictions, feed_dict={self.x_placeholder: x , self.kp_placeholder: 1.0})
    
    def mlp(self, x, keep_prob):
        activation_list = {
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'relu': tf.nn.relu}
        
        layer_1 = tf_layer(x, int(self.nodes_hidden[0]), activation_fn=activation_list[self.activation_name[0]])
        layer_2 = tf_layer(layer_1, int(self.nodes_hidden[1]), activation_fn=activation_list[self.activation_name[1]])
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        layer_3 = tf_layer(layer_2, 1, activation_fn=None)
        
        return layer_3
    
    
    def loss_dnn(self, y_hat ,y ):
        
        return tf.reduce_mean(tf.square(tf.subtract(y_hat, y)))

        
    def save(self):
        
        return 1
    
    def load(self):
        
        return 1
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
    

#     def minimize_func(self, optimal, sig, i=0):
#         if i==0:
#             optimal = self.X[np.random.randint(self.X.shape[0])]
#         def funct(x):    
#             const =  self.constraint(x)
#             f = 0 
#             for i in range(len(const)):
#                 f += sig*max(0.0, const[i]**2)
#             return self.kr_func(x) + f
        
#         res = minimize(funct, optimal, method='nelder-mead', options={'xtol': 1e-16, 'disp': False, 'maxiter': 1000})
#         optimal = res.x
        
#         return optimal
