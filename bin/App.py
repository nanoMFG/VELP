#!/usr/bin/env python
# coding: utf-8

# # VELP
# Author: Alireza Moradzadeh 
# 
# contact: moradza2@illinois.edu 
# 
# Nano-ManuFacturing Group (NMFG) 
# 
# Department of Mechanical Science and Engineering (MechSE) 
# 
# University of Illinois at Urbana-Champaign (UIUC) 
# 
# Version = 1.0
# 

# In[1]:


import tkinter as tk               
from tkinter import font  as tkfont
from tkinter import *
import tkinter.messagebox as tkMessageBox
import tkinter.filedialog as tkFileDialog
import tkinter.ttk as ttk
import os
import PIL.Image 
import PIL.ImageTk
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as Kernels  
from scipy.optimize import minimize
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as tf_layer

global All_material

def read_data_global(path='data/data.dat'):
        cnt = 0
        with open(path, 'r') as f:
            data = []
            file = f.read().split('\n')
            for line in file:
                if cnt == 0:
                    All_material = line.split('\t')[:-1]
                else: 
                    data.append([float(a) for a in  line.split('\t')])
                cnt += 1
        data = np.array(data)
#         print("Done with reading the dataset!")
#         print("Material List:")
#         print(All_material)
#         print("Numbers of data point:", cnt)
        
        return All_material, data[:,:-1], data[:,-1].reshape((-1,1))
All_material, _, _ = read_data_global()


class Kernel_Optimization():
    def __init__(self, dict_mat=None, kernel_type='RBF', CV=5):
        self._kernel_type = kernel_type
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
        
        All_material, self.X , self.y = self.read_data()

        self.kr.fit(self.X, self.y)
        
    def kr_func(self, x):
        return self.kr.predict(x)
    
    def read_data(self,path='data/data.dat'):
        cnt = 0
        with open(path, 'r') as f:
            data = []
            file = f.read().split('\n')
            for line in file:
                if cnt == 0:
                    All_material = line.split('\t')[:-1]
                else: 
                    data.append([float(a) for a in  line.split('\t')])
                cnt += 1
        data = np.array(data)        
        return All_material, data[:,:-1], data[:,-1].reshape((-1,1))
    
    def constraint(self, x):    
        ''' Create Constraints for physically-consistent solvent decomposition
            sum_cat x_i = 1.0 & sum_an x_i = 1.0 , x_i > 0 for both cation and anaion
        '''
        
        n_cations = 0
        n_anions = 0
        
        for k in All_material:
            if k[-1] =='+':
                n_cations += 1
            else:
                n_anions += 1
        n_constraints = len(All_material)+ 2
        
        for cnt, m in enumerate(All_material):
            if self.dict_mat[m]:
                n_constraints -= 1
            if x[cnt] <0 or x[cnt] > 1:
                n_constraints += 1
        val_constraints =  np.zeros((n_constraints))
        
        cat_list = []
        an_list = []
        # constraints 
        for k, v in self.dict_mat.items():
            if v and k[-1]=='+':
                cat_list.append(k)
            if v and k[-1]=='-':
                an_list.append(k)
                
        cnt = 2
        for i in range(len(All_material)):
            if All_material[i] in cat_list:
                val_constraints[0] += x[i]
            elif All_material[i] in an_list:
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

class DNN():
    def __init__(self, dict_mat=None, n_dims=10, dnn_features=[20,10, 0.75, 0.001, 1000000], kernel_type='DNN'):
        tf.reset_default_graph()
        self.dict_mat = dict_mat
        self._kernel_type = kernel_type
        
        # network width
        self.nodes_hidden = dnn_features[:2]
        
        # keep_prob, learning rate
        self.keep_prob = dnn_features[2]
        self.lr = dnn_features[3]
        
        # num of training step and feature dimension
        self.num_step = dnn_features[4]
        self.n_dims = n_dims
        
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
        All_material, self.X , self.y = self.read_data()
        
        # minimization 
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder).minimize(self.loss)
        
        # Create Session and Initialize Variables
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        
    def kr_func(self, x):
        x = x.reshape((-1, len(All_material)))
        return self.session.run(self.predictions, feed_dict={self.x_placeholder: x , self.kp_placeholder: 1.0})
    
    def mlp(self, x, keep_prob):
        layer_1 = tf_layer(x, int(self.nodes_hidden[0]), activation_fn=tf.nn.sigmoid)
        layer_2 = tf_layer(layer_1, int(self.nodes_hidden[1]), activation_fn=tf.nn.sigmoid)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        layer_3 = tf_layer(layer_2, 1, activation_fn=None)
        
        return layer_3
    
    def loss_dnn(self, y_hat ,y ):
        
        return tf.reduce_mean(tf.square(tf.subtract(y_hat, y)))

        
    def save(self):
        
        return 1
    
    def load(self):
        
        return 1
    
    def read_data(self,path='data/data.dat'):
        cnt = 0
        with open(path, 'r') as f:
            data = []
            file = f.read().split('\n')
            for line in file:
                if cnt == 0:
                    All_material = line.split('\t')[:-1]
                else: 
                    data.append([float(a) for a in  line.split('\t')])
                cnt += 1
        data = np.array(data)
        print("Done with reading the dataset!")
        print("Material List:")
        print(All_material)
        print("Numbers of data point:", cnt)
        
        return All_material, data[:,:-1], data[:,-1].reshape((-1,1))
    
    def constraint(self, x):    
        ''' Create Constraints for physically-consistent solvent decomposition
            sum_cat x_i = 1.0 & sum_an x_i = 1.0 , x_i > 0 for both cation and anaion
        '''
        
        n_cations = 0
        n_anions = 0
        
        for k in All_material:
            if k[-1] =='+':
                n_cations += 1
            else:
                n_anions += 1
        n_constraints = len(All_material)+ 2
        
        for cnt, m in enumerate(All_material):
            if self.dict_mat[m]:
                n_constraints -= 1
            if x[cnt] <0 or x[cnt] > 1:
                n_constraints += 1
        val_constraints =  np.zeros((n_constraints))
        
        cat_list = []
        an_list = []
        for k, v in self.dict_mat.items():
            if v and k[-1]=='+':
                cat_list.append(k)
            if v and k[-1]=='-':
                an_list.append(k)
        cnt = 2
        for i in range(len(All_material)):
            if All_material[i] in cat_list:
                val_constraints[0] += x[i]
            elif All_material[i] in an_list:
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

    def minimize_func(self, optimal, sig, i=0):
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
    
class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.__version__ = "1.0"
        self.dict_mat={'Py+':tk.IntVar(), 'BPy+': tk.IntVar(), 'DCA-': tk.IntVar(),                        'PF6-': tk.IntVar(), 'TF-': tk.IntVar(), 'TFSI-': tk.IntVar()}
        
        self.title_font = tkfont.Font(family='Arial', size=18, weight="bold", slant="italic")
        self.text_font = tkfont.Font(family='Arial', size=12)
        self.title("VELP")
        self.configure(borderwidth="1")
        self.geometry("650x750")
        self.wm_iconbitmap('E:\Research Thesis\Paper\Exfoliation\Reports\icon.jpeg.ico')
        # the container has various frames stacked, selected frame will be raised 
        # above the other frames 
        
        container = tk.Frame(self, width=300, height=300, bg="", colormap="new")
        container.pack()
        
        self.k_method = None
        
        container.grid_rowconfigure(10, weight=1)
        container.grid_columnconfigure(10, weight=1)

        self.frames = {}
        
        
        for F in (PageStart, PageDocumentary, PageMaterials, PageMachineLearning, PageDesign):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        def refresh():
            self.destroy()
            App().mainloop()
            
        def donothing():
            print("Not implemented!")
            
        self.menubar=tk.Menu(self)
        self.filemenu=Menu(self.menubar,tearoff=0)
        self.filemenu.add_command(label="New", command=refresh)
        self.filemenu.add_command(label="Save", command=donothing, state="disabled")
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.destroy)
        
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        
        self.helpmenu=tk.Menu(self.menubar,tearoff=0)
        self.helpmenu.add_command(label="Help",command=self.show_help)
        self.menubar.add_cascade(label="Help",menu=self.helpmenu)
        self.config(menu=self.menubar)
        
        self.show_frame("PageStart")
        
    def popup_warning(self):
        win = tk.Toplevel()
        win.wm_title("Warning")
        win.wm_iconbitmap('E:\Research Thesis\Paper\Exfoliation\Reports\icon.jpeg.ico')
        l = tk.Label(win, text="Warning: The system is not neutral, please reselect cations and anions!")
        l.grid(row=0, column=0)
        self.grab_set()
        b = ttk.Button(win, text="Okay", command=win.destroy)
        b.grid(row=1, column=0)
        
    def show_help(self):
        win = tk.Toplevel()
        win.wm_title("Help")
        l = tk.Label(win, text="VELP allows user to combie molecular dynamics simulations results with \n machine learning algorithms (selected by user) to find optimal solvent \n for exfolation in liquid phase of 2D materials or to predict relative \n performance of a specific solvent content!\n", font=tkfont.Font(family='Arial', size=12))
        l.pack()
        self.grab_set()
        b = ttk.Button(win, text="Okay", command=win.destroy)
        b.pack()
        
    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
    
        
    def get_page(self, page_class):
        return self.frames[page_class]

class PageStart(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Welcome to VELP", font=controller.title_font)
        label.grid(row=0, column=0)
        
        label2 = tk.Label(self, text="Developed at the University of Illinois at Urbana-Champaing",                            font=controller.text_font)
        label2.grid(row=1, column=0)
        
        label3 = tk.Label(self, text="Department of Mechanical Science and Engineering",                            font=controller.text_font)
        label3.grid(row=2, column=0)
        
        label4 = tk.Label(self, text="Nano-Manufacturing Group",                            font=controller.text_font)
        label4.grid(row=3, column=0)
        
        button1 = tk.Button(self, text="Go to Documentary", 
                            command=lambda: controller.show_frame("PageDocumentary"),font=controller.text_font, width=20)
        button2 = tk.Button(self, text="Material Selection",
                            command=lambda: controller.show_frame("PageMaterials"), font=controller.text_font, width=20)
        button1.grid(row=4, columnspan=1, column=0)
        button2.grid(row=5, columnspan=1, column=0)
        
        ref = PIL.Image.open('E:\Research Thesis\Paper\Exfoliation\Reports\Exfoliationof2DMaterials.png')
        photo = PIL.ImageTk.PhotoImage(ref)
        first_img = tk.Label(self, image=photo)
        first_img.image = photo
        first_img.grid(row=6, column=0)
        

class PageDocumentary(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        button = tk.Button(self, text="Go to the start menu",                           command=lambda: controller.show_frame("PageStart"), font=controller.text_font )
        button.pack()
        label = tk.Label(self, text="Documentary", font=controller.title_font)
        label.pack()
        
        docs = "Virtual Exfoliation in Liquid Phase(VELP):\n             Based on the dataset and user defined materials, VELP employes machine learning \n algorithms to screen for optimal solvent for exfoliation process. \n Currently, VELP uses potential of mean force as the selection criterion!            \n VELP is pushing the boundaries of Exfoliation Process Solvent Design!"
        
        label_doc = tk.Label(self,text=docs,font=controller.text_font)
        label_doc.pack()
        
        label = tk.Label(self, text="Developers: Alireza Moradzadeh and Narayana Aluru", font=controller.text_font)
        label.pack()
        
        label = tk.Label(self, text="Contact: moradza2@illinois.edu, aluru@illinois.edu", font=controller.text_font)
        label.pack()
        
        ref = PIL.Image.open('E:\Research Thesis\Paper\Exfoliation\Reports\icon.jpg')
        photo = PIL.ImageTk.PhotoImage(ref)
        first_img = tk.Label(self, image=photo)
        first_img.image = photo
        first_img.pack(fill='x')
        

class PageMaterials(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Materials Selection", font=controller.title_font)
        label.pack(side="top")
        
        button = tk.Button(self, text="Go to the start page",                           command=lambda: controller.show_frame("PageStart"), font=controller.text_font)
        button.pack()
        
        All_material = ["Py+", 'BPy+','DCA-', 'PF6-', 'TF-', 'TFSI-']
        
        label = tk.Label(self, text="Cation and Anion ", font=controller.text_font)
        label.pack(side=TOP)
        label = tk.Label(self, text="", font=controller.text_font)
        label.pack(side=TOP)
        
        self.List_of_CB = [ ]
        
        self.dict_mat2={"Py+":tk.IntVar(), 'BPy+': tk.IntVar(), 'DCA-': tk.IntVar(),                        'PF6-': tk.IntVar(), 'TF-': tk.IntVar(), 'TFSI-': tk.IntVar()}
        
        for mat in All_material:
            self.List_of_CB.append(tk.Checkbutton(self, text=mat, variable=self.controller.dict_mat[mat]))
        cnt_an = 3
        cnt_cat = 3
        for n, l in enumerate(self.List_of_CB):
            l.pack(side=TOP, anchor=W,in_=label)
            l.config(font=controller.text_font)
        
        button = tk.Button(self, text="Materials are selected!",                           command=self.set_dict, font=controller.text_font)
        button.pack()
        
        button = tk.Button(self, text="back",
                           command=lambda: controller.show_frame("PageStart"), font=controller.text_font, width=4)
        button.pack()
        
        button = tk.Button(self, text="next",                           command=lambda: controller.show_frame("PageMachineLearning"), font=controller.text_font, width=4)
        button.pack()
        
    def set_dict(self):
        n_cat = 0
        n_an = 0
        for mat in All_material:
            self.controller.dict_mat[mat].get()
            self.dict_mat2[mat].set(self.controller.dict_mat[mat].get())
            if self.controller.dict_mat[mat].get() and mat[-1]=='+':
                n_cat +=1
            elif self.controller.dict_mat[mat].get() and mat[-1]=='-':
                n_an +=1
                
        if n_an ==0  or n_cat ==0:
            self.controller.popup_warning()
                
class PageMachineLearning(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Method Selection", font=controller.title_font)
        label.pack(side="top", fill="x", pady=12)
        
        button = tk.Button(self, text="Go to the start page",                           command=lambda: controller.show_frame("PageStart"),  font=controller.text_font)
        
        Kernels = ['RBF', 'ExpSineSquared']
        
        label = tk.Label(self, text="Methods", font=controller.text_font)
        label.pack(side="top", fill="x", pady=12)
        
        self.kernel_type = tk.StringVar(self)
        self.kernel_type.set("RBF") # default value
        
        w = tk.OptionMenu(self, self.kernel_type, "RBF", "ExpSineSquared","DNN")
        w.pack()
        w.config(font=controller.text_font)
        
        button = tk.Button(self, text="Show selected materials",                           command= self.show_mat, font=controller.text_font)
        button.pack()
        
        self.selected_mat = tk.Label(self, text="", font=controller.text_font)
        self.selected_mat.pack()
        
        button = tk.Button(self, text="Method selected",                           command=self.show_method,font=controller.text_font, takefocus=False)
        button.pack()
        
        button = tk.Button(self, text="back",
                           command=lambda: controller.show_frame("PageMaterials"), font=controller.text_font)
        button.pack() 
        
        button = tk.Button(self, text="next",
                           command=lambda: controller.show_frame("PageDesign"), font=controller.text_font)
        button.pack() 
        
    def show_mat(self):
        text = []
        for mat in All_material:
            if self.controller.dict_mat[mat].get():
                text.extend(mat)
                text.extend(',')
                
        self.selected_mat.config(text=text[:-1])
        self.selected_mat.pack()
        
    def show_method(self):
        dict_mat = {}
        for mat in All_material:
            v = self.controller.dict_mat[mat].get()
            dict_mat.update({mat: v})   
            
        kernel_type = self.kernel_type.get()
        
        if kernel_type =='DNN':
            
            win = tk.Toplevel(self)
            win.wm_title("DNN")
            win.wm_iconbitmap('E:\Research Thesis\Paper\Exfoliation\Reports\icon.jpeg.ico')

            l = tk.Label(win, text="Deep Neural Network",font=self.controller.title_font)
            l.grid(row=0, columnspan=2)
            
            self.grab_set()
            
            l = tk.Label(win, text="Number of nodes in layer 1: ",font=self.controller.text_font)
            l.grid(row=2, column=0,sticky='W')
            
            n_hidden_layer_1 = tk.Entry(win)
            n_hidden_layer_1.insert(END, "10")
            n_hidden_layer_1.grid(row=2, column=1)
              
            l = tk.Label(win, text="Number of nodes in layer 2: ",font=self.controller.text_font)
            l.grid(row=4, column=0, sticky='W')
            
            n_hidden_layer_2 = tk.Entry(win)
            n_hidden_layer_2.insert(END, "6")
            n_hidden_layer_2.grid(row=4, column=1)
            
            l = tk.Label(win, text="Keep Probability [0, 1]: ",font=self.controller.text_font)
            l.grid(row=6, column=0, sticky='W')
            
            keep_prob = tk.Entry(win)
            keep_prob.insert(END,"0.75")
            keep_prob.grid(row=6, column=1)
            
            l = tk.Label(win, text="Learning rate :",font=self.controller.text_font)
            l.grid(row=8, column=0,sticky='W')
            
            lr_entry = tk.Entry(win)
            lr_entry.insert(END, "0.0001")
            lr_entry.grid(row=8, column=1)
            
            l = tk.Label(win, text="Number of training steps :",font=self.controller.text_font)
            l.grid(row=10, column=0,sticky='W')
            
            num_step_entry = tk.Entry(win)
            num_step_entry.insert(END, "10000")
            num_step_entry.grid(row=10, column=1)
            
            l = tk.Label(win, text="Fraction of training (0, 1] :",font=self.controller.text_font)
            l.grid(row=12, column=0,sticky='W')
            
            frac_entry = tk.Entry(win)
            frac_entry.insert(END, "0.75")
            frac_entry.grid(row=12, column=1)
            
            train_l = tk.Label(win, text="Training loss",font=self.controller.text_font)
            train_l.grid(row=14, column=0,sticky='W')
            
            train_l = tk.Label(win, text="", font=self.controller.text_font)
            train_l.grid(row=16, column=0,sticky='W')
            
            valid_l = tk.Label(win, text="Validation loss",font=self.controller.text_font)
            valid_l.grid(row=14, column=1,sticky='W')
            
            valid_l = tk.Label(win, text="",font=self.controller.text_font)
            valid_l.grid(row=16, column=1,sticky='W')
            
            
            global dnn_feature
            
            dnn_feature = [10 , 5, 0.75, 0.0001, 1000000] # nodes 1, nodes 2, keep_prob , learning rate, training steps
            
            def get_destroy():
                
                dnn_feature[0] = int(n_hidden_layer_1.get())
                dnn_feature[1] = int(n_hidden_layer_2.get())
                dnn_feature[2] = float(keep_prob.get())
                
                if dnn_feature[2] < 0.0 or dnn_feature[2] > 1.0:
                    raise ValueError("keep_prob should be [0, 1]")
                
                dnn_feature[3] = float(lr_entry.get())
                
                
                dnn_feature[4] = int(num_step_entry.get())
                tf.reset_default_graph()
                self.controller.k_method = DNN(dict_mat=dict_mat, n_dims=len(All_material), dnn_features=dnn_feature)
                
                if self.controller.k_method.session._opened:
                    self.controller.k_method.session.close()
                
                if float(frac_entry.get()) <= 0.0 or float(frac_entry.get()) > 1.0:
                    raise ValueError("keep_prob should be [0, 1]")
                
                frac = int(self.controller.k_method.X.shape[0]*float(frac_entry.get()))
                
                for i in range(int(dnn_feature[4])):
                    feed = {self.controller.k_method.x_placeholder: self.controller.k_method.X[:frac],                            self.controller.k_method.y_placeholder: self.controller.k_method.y[:frac],                             self.controller.k_method.kp_placeholder: dnn_feature[2], self.controller.k_method.lr_placeholder: dnn_feature[3]}
                    self.controller.k_method.session.run(self.controller.k_method.train_opt, feed_dict=feed)
                    
                    feed = {self.controller.k_method.x_placeholder: self.controller.k_method.X[:frac], self.controller.k_method.y_placeholder:                             self.controller.k_method.y[:frac], self.controller.k_method.kp_placeholder: 1.0}
                    feed_v = {self.controller.k_method.x_placeholder: self.controller.k_method.X[frac:],                              self.controller.k_method.y_placeholder: self.controller.k_method.y[frac:],                              self.controller.k_method.kp_placeholder: 1.0}
                    
                    train_loss = self.controller.k_method.session.run(self.controller.k_method.loss, feed_dict=feed)
                    valid_loss = self.controller.k_method.session.run(self.controller.k_method.loss, feed_dict=feed_v)
            
                    train_l.config(text=str(round(train_loss,5)))
                    valid_l.config(text=str(round(valid_loss,5)))
                    
                    train_l.update()
                    valid_l.update()
                    
                self.grab_release()
                    
            b = ttk.Button(win, text="train", command=get_destroy)
            b.grid(row=18, column=0, sticky=W)
            
            kill_b = ttk.Button(win, text="done", command=win.destroy)
            kill_b.grid(row=18, column=1, sticky=E)
            
        else:
            win = tk.Toplevel(self)
            win.wm_title(kernel_type)
            win.wm_iconbitmap('E:\Research Thesis\Paper\Exfoliation\Reports\icon.jpeg.ico')
            
            l = tk.Label(win, text="Kernels",font=self.controller.title_font)
            l.grid(row=0, columnspan=2)
            
            self.grab_set()
            
            l = tk.Label(win, text="k-fold cross validation : ",font=self.controller.text_font)
            l.grid(row=2, column=0,sticky='W')
            
            CV = tk.StringVar(win)
            
            CV.set("5") # default value
            
            w = tk.OptionMenu(win, CV, "1", "2", "3", "4", "5","6","7","8","9","10")
            w.grid(row=2, column=1)
            
            def get_destroy():
                b.config(text='Fitting Kernel')
                b.update()
                b.grid(row=4, columnspan=2)
                self.controller.k_method = Kernel_Optimization(dict_mat=dict_mat, kernel_type=kernel_type, CV=int(CV.get()))
                self.grab_release()
                win.destroy()
            
            b = ttk.Button(win, text="Done", command=get_destroy)
            b.grid(row=4, columnspan=2)  
            
class PageDesign(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Solvent Design", font=controller.title_font)
        label.pack(side="top", fill="x", pady=12)
    
        button = tk.Button(self, text="Go to the start page",                           command=lambda: controller.show_frame("PageStart"),  font=controller.text_font)
        
        button = tk.Button(self, text="Show selected materials",                           command= self.show_mat, font=controller.text_font)
        button.pack()
        
        self.selected_mat = tk.Label(self, text="", font=controller.text_font)
        self.selected_mat.pack()
        
        button = tk.Button(self, text="Optimal Solvent",                           command=self.show_optimal,font=controller.text_font, takefocus=False)
        button.pack()
        
        self.prog_v = tk.StringVar()
        self.prog_v.set("Progress : 0 ")
        
        self.progress = tk.Label(self, textvariable=self.prog_v, font=controller.text_font)
        self.progress.pack()
        
        self.result = tk.Label(self, text=" ", font=controller.text_font)
        self.result.pack()
        self.optimal = None
        
        button = tk.Button(self, text="Predict Solvent Perofrmance",                           command=self.show_prediction,font=controller.text_font)
        button.pack()
        
        self.predict = tk.Label(self, text=" ", font=controller.text_font)
        self.predict.pack()
#         self.controller.filemenu.add_command(label="Save", command=self.save_result, state="active")
        
        button = tk.Button(self, text="save",
                           command=self.save_result, font=controller.text_font)
        button.pack() 
        
        button = tk.Button(self, text="back",
                           command=lambda: controller.show_frame("PageMachineLearning"), font=controller.text_font)
        button.pack() 
        
        button = tk.Button(self, text="start menu",
                           command=lambda: controller.show_frame("PageStart"), font=controller.text_font)
        button.pack() 
        
    def show_mat(self):
        text = []
        for mat in All_material:
            if self.controller.dict_mat[mat].get() :
                text.extend(mat)
                text.extend(',')
                
        self.selected_mat.config(text=text[:-1])
        self.selected_mat.pack()
        
    def show_optimal(self):        
        for i, sig in enumerate(np.logspace(-2,47,50)):
            self.optimal = self.controller.k_method.minimize_func(self.optimal, sig, i)
            
            percent_b = 2*(i+1)
            progress_b = 'Progress : '+ str(percent_b)
            
            self.prog_v.set(progress_b)
            self.progress.update()
        
            result = []
            for i, mat in enumerate(All_material):
                if self.controller.dict_mat[mat].get():
                    result.extend(str(np.round(self.optimal[i],3)))
                    result.extend(',')
            
            self.result.config(text=result[:-1])
            self.result.pack()
            self.result.update()
            
    def show_prediction(self):
        win = tk.Toplevel(self)
        win.wm_title("Enter Solvent Composition")
        win.wm_iconbitmap('E:\Research Thesis\Paper\Exfoliation\Reports\icon.jpeg.ico')
            
        l = tk.Label(win, text="Solvent Composition",font=self.controller.title_font)
        l.grid(row=0, columnspan=2)
        
        warning_l = tk.Label(win, text="",font=self.controller.title_font)
        warning_l.grid(row=1, columnspan=2)
                
        self.grab_set()
        x = np.zeros((1,len(All_material)))
        x_l_entries = []
        for i in range(len(All_material)):
            x_l = tk.Label(win, text=All_material[i],font=self.controller.text_font)
            x_l.grid(row=2*i+2, column=0,sticky='W')
            x_l_entries.append(tk.Entry(win))
            x_l_entries[i].insert(END, "0.00")
            x_l_entries[i].grid(row=2*i+2, column=1,sticky='W')
       
        def get_destroy():
            self.grab_set()
            
            for i in range(len(All_material)):
                x[0,i] =  float(x_l_entries[i].get())
            
            x_cat = 0.0
            x_an = 0.0
            for i in range(len(All_material)):
                if All_material[i][-1] == '+':
                    x_cat += x[0,i]
                elif All_material[i][-1] == '-':
                    x_an += x[0,i]
            
            if not x_cat == 1.0 or not x_an == 1.0:
                warning_l.config(text="System is not neutral!", foreground='red')
                warning_l.update()
                
            result= self.controller.k_method.kr_func(x)
            self.predict.config(text=str(round(result[0,0],3)))
            self.predict.pack()
            self.grab_release()
            
        b = ttk.Button(win, text="Insert Data", command=get_destroy)
        b.grid(row=2*len(All_material)+4, column=0)   
        
        b = ttk.Button(win, text="Done", command=win.destroy)
        b.grid(row=2*len(All_material)+4, column=1)   
        
    def save_result(self, path='result',name='result.dat'):        
        if len(self.optimal) == 0:
            raise NameError('Design is not performed!')
        
        if not os.path.exists(path):
            os.makedirs('result')
            
        with open(os.path.join(path, name),'w') as f:
            for n in range(2):
                if n == 0:
                    for mat in All_material:
                        f.write(mat+'\t')
                    f.write('\n')
                else:
                    for x_solv in self.optimal:
                        f.write(str(np.round(x_solv,3))+'\t')

if __name__ == "__main__":
    app = App()
    print("VELP Version: ", app.__version__)
    app.mainloop()


# In[ ]:





# In[ ]:


if not keep_p:
    print(10)


# In[ ]:


print(type(keep_p))


# In[ ]:




