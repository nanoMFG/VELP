import ipywidgets as widgets
from hublib.ui import FileUpload, PathSelector, Download
import hublib.ui as ui
import os
import re
import sys
from IPython.display import display, clear_output
from ipywidgets import Button, Layout
from mavelp.kernel_methods import *
from mavelp.neural_network import *
from mavelp.data_tool import *
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

MAVELP_DATA_PATH="./data/data.dat"
MAVELP_DATA_PATH=os.environ.get("MAVELP_DATA_PATH")

Materials, Features, Labels = read_data_global(MAVELP_DATA_PATH)

class tabs():
    
    def __init__(self, tablist= ['Material Selction', 'Method Selection', 'Design', 'MA-VELP'], Materails=['BPY', 'TF']):
        """
        Initialize a tab with multiple childrens (len(tablist))
        It will generate each tab with its own function. Ensure that functions are defined.
        
        """
        if not os.path.exists(MAVELP_DATA_DIR):
            print(f"Data diretory:{MAVELP_DATA_DIR} does not exist")
            raise FileNotFoundError

        self.tablist = tablist
        self.tab = widgets.Tab()
        
        
        self.tab.children = ()
        
        self.Materials_List = ['BPY', 'PY','PF6','TF','TFSI','DCA']
        self.Materials = Materails
        
        for name in self.tablist:
            if name == 'MA-VELP':
                self.tab.children += (self.generate_docs(),)
            elif name == 'Material Selction':
                self.tab.children += (self.generate_materials(),)
            elif name == 'Method Selection':
                self.tab.children += (self.generate_method(), )
            elif name == 'Design':
                self.tab.children += (self.generate_design(),)
            else:
                raise NameError('Wrong Tab name inserted: ('+ name+')')
                
        for cnt, name in enumerate(self.tablist):
            self.tab.set_title(cnt, name)
        
        display(self.tab)
    
    def generate_docs(self):
         
        discription_text = "Machine-Learned Assisted Virtual Exfoliation in Liquid Phase(MA-VELP):\n\n\n \n\
        Based on the dataset obtained from high-throughput computational study, MAVELP employes machine learning algorithms to screen for optimal solvent based on the user's material selection for exfoliation process via liquid phase. Currently, MAVELP uses potential of mean force as the selection criterion! MAVELP developed at University of Illinois at Urbana-Champaign under Nano-Manufacturing Group is pushing the boundaries of Exfoliation Process Solvent Design! \
            \n \n \n \n \n \nFor assitant contact: moradza2@illinois.edu"
            
        output =  widgets.Output()
        with output:
            print(discription_text)
            print("\nDevelopers: Alireza Moradzadeh, Darren Adams, and Narayana R. Aluru")
                
            
            
        return widgets.VBox([output], layout={'height': '400px'})
    
    
    def generate_method(self):
        style = {'description_width': 'initial'}
        style2 = {'description_width': '150px'}
        self.method = widgets.Dropdown(
            options=[('Neural Network', 1), ('Kernel Ridge Regression', 2)],
            value=2,
            description='Method:', style=style, layout={'width': 'max-content'})
        self.submit_buttom = widgets.Button(description="Run regression")
        self.progress = widgets.FloatProgress(
                    value=0.0,
                    min=0,
                    max=10.0,
                    step=0.1,
                    description='Progress:',
                    bar_style='info',
                    orientation='horizontal',visibility = 'hidden', style=style, layout={'width': '250px'})
        
        self.progress.layout.visibility = 'hidden'
        
        output = widgets.Output()#layout={'width': '300px', 'height': '300px'})
        def on_button_clicked(b):
            
            output.clear_output()
            if self.method.value == 2:
                self.progress.layout.visibility = 'hidden'
                with output:
                    print("Starting to fit!")
                
                self.ml = Kernel_Optimization(dict_mat=self.dict_mat, kernel_type=self.method_options_kr_type.value , CV=self.method_options_kr_cv.value, X=Features, y=Labels, All_material=Materials)
                
                with output:
                    print("Kernel is fitted!")
                    R2 = self.ml.kr.score(Features,Labels )
                    print("R^2: ", np.round(R2))
            elif self.method.value == 1:
                self.progress.value = 0    
                self.progress.layout.visibility = 'visible'
                
                self.ml = DNN(n_dims=6, dnn_features=[self.methods_options_dnn_h1.value, self.methods_options_dnn_h2.value,\
                                                  self.methods_options_dnn_kp.value, self.methods_options_dnn_lr.value,\
                                                  self.method_options_dnn_nsteps.value],\
                                                  activation_name=[self.methods_options_dnn_h1_f.value, self.methods_options_dnn_h2_f.value], X=Features,y=Labels )
                num_shows = int(self.method_options_dnn_nsteps.value/self.method_options_dnn_show_nsteps.value)
                cur_shows = 0
                Losses = np.zeros((num_shows))
                
                
                for epoch in range(self.method_options_dnn_nsteps.value):
                    
                    feed = {self.ml.x_placeholder: self.ml.X, self.ml.y_placeholder: self.ml.y,\
                            self.ml.kp_placeholder:self.methods_options_dnn_kp.value, self.ml.lr_placeholder: self.methods_options_dnn_lr.value}
                    self.ml.session.run(self.ml.train_opt, feed_dict=feed)
                    k = (10*(epoch+1)/self.method_options_dnn_nsteps.value)
                    self.progress.value = k
                    
                    if epoch % self.method_options_dnn_show_nsteps.value == 0 :
                        Losses[cur_shows] = self.ml.session.run(self.ml.loss, feed)
                        cur_shows += 1
                with output:
                    fig = plt.figure(1, figsize=(6,5))
                    plt.plot(self.method_options_dnn_show_nsteps.value*np.arange(num_shows), Losses,'r-o')
                    plt.show()
                    self.savefig_button = widgets.Button(description="Save Figure")
                    self.savefig_name = widgets.Text(value='Result',
                                        placeholder='Type something',
                                        description='File name:',
                                        disabled=False)
                    def save_fig_on_button_clicked(c):
                        fig = plt.figure(1, figsize=(6,5))
                        plt.plot(self.method_options_dnn_show_nsteps.value*np.arange(num_shows), Losses,'r-o')
                        plt.savefig(self.savefig_name.value+'.jpeg', dpi=150)
                        plt.close(fig)
                    self.savefig_button.on_click(save_fig_on_button_clicked)
                    display(self.savefig_button)
                    display(self.savefig_name)
#                     self.savemodel_button = widgets.Button(description="Save Model")
                    
                    
            elif self.method.value == 3:
                
                with output:
                    print("This method is not implemented yet!")
                
                

        self.submit_buttom.on_click(on_button_clicked)
        
        Method_Box = widgets.VBox([self.method, self.submit_buttom , self.progress, output], layout={'width': '550px'})
        # Kernel Rigde Regression Options
        self.method_options_kr_cv = widgets.IntSlider(value=2, min=2,max=10, step=1,
            description='Cross-Validation:',
            disabled=False, continuous_update=False,
            orientation='horizontal',readout=True,
            readout_format='d', style=style)
        
        self.method_options_kr_type = widgets.Dropdown(
            options=['RBF', 'ExpSineSquared'],
            value='RBF',
            description='Kernel Type:', style={'description_width': '100px'}, layout={'width': '260px'})
        
        # Neural Network Option
        
        self.methods_options_dnn_h1 = widgets.IntSlider(value=15, min=1,max=100, step=1,
            description='Hidden Layer 1 Nodes:',
            disabled=False, continuous_update=False,
            orientation='horizontal',readout=True,
            readout_format='d', style=style2)#,  layout={'width': '260px'})
        
        self.methods_options_dnn_h2 = widgets.IntSlider(value=5, min=1,max=50, step=1,
            description='Hidden Layer 2 Nodes:',
            disabled=False, continuous_update=False,
            orientation='horizontal',readout=True,
            readout_format='d', style=style2)#, layout={'width': '260px'})
        
        self.methods_options_dnn_h1_f = widgets.Dropdown(
            options=['tanh','sigmoid','relu'],
            value='tanh',
            description='Activation Layer 1:', style=style2, layout={'width': '260px'})
        
        self.methods_options_dnn_h2_f = widgets.Dropdown(
            options=['tanh','sigmoid','relu'],
            value='tanh',
            description='Activation Layer 2:', style=style2, layout={'width': '260px'})
        self.methods_options_dnn_loss = widgets.Dropdown(
            options=[('MSE', 1), ('MAE', 2)],
            value=1,
            description='Loss function: ', style=style2, layout={'width': '260px'})
        self.methods_options_dnn_optimizer = widgets.Dropdown(
            options=[('Adagrad', 1), ('SGD', 2), ('Adam', 3)],
            value=1,
            description='Optimizer: ', style=style2, layout={'width': '260px'})
        
        self.methods_options_dnn_lr = widgets.BoundedFloatText(
                value=.0001,
                min=0,
                max=1.0,
                step=0.000001,
                description='Learning Rate:',
                disabled=False, style=style2, layout={'width': '260px'})
        
        self.methods_options_dnn_kp = widgets.BoundedFloatText(
                value=0.001,
                min=0,
                max=1.0,
                step=0.001,
                description='Keep Probability:',
                disabled=False, style=style2, layout={'width': '260px'})
        
        
        self.method_options_dnn_nsteps = widgets.BoundedIntText(value=5000,step=100,min=1000, max = 1e7,
            description='Training steps:',
            disabled=False, continuous_update=False,
            orientation='horizontal',readout=True,
            readout_format='d', style=style2)
        
        self.method_options_dnn_show_nsteps = widgets.BoundedIntText(value=1000,step=10,
            description='Show steps:',
            min =10, max= 100000,
            disabled=False, continuous_update=False,
            orientation='horizontal',readout=True,
            readout_format='d', style=style2)
        
        
        # Random Forest Option
        self.method_options_rf_max_depth = widgets.IntSlider(value=3, min=1,max=10, step=1,
            description='Depth of Tree:',
            disabled=False, continuous_update=False,
            orientation='horizontal',readout=True,
            readout_format='d', style=style2)
        
        self.method_option_rf_n_estimators = widgets.BoundedIntText(
                value=100,
                min=1,
                max=500,
                step=1,
                description='Num. Estimators:',
                disabled=False, style=style2, layout={'width': '260px'})
        
        
        
        V_box_KR = widgets.VBox([self.method_options_kr_cv,self.method_options_kr_type], layout={'width': '325px'})
        V_box_DL = widgets.VBox([self.methods_options_dnn_h1, self.methods_options_dnn_h2, self.methods_options_dnn_h1_f, self.methods_options_dnn_h2_f, \
                                 self.methods_options_dnn_loss,self.methods_options_dnn_optimizer, self.methods_options_dnn_lr, self.method_options_dnn_nsteps,\
                                 self.method_options_dnn_show_nsteps, self.methods_options_dnn_kp], layout={'width': '400px'})
        V_box_RF = widgets.VBox([self.method_options_rf_max_depth, self.method_option_rf_n_estimators], layout={'width': '450px'})
        
        V_box_DL.layout.visibility = 'hidden'
        V_box_KR.layout.visibility = 'visible'
        V_box_RF.layout.visibility = 'hidden'
        
        
        def on_selection_button(b):
            if self.method.value == 1:
                V_box_DL.layout.visibility = 'visible'
                V_box_RF.layout.visibility = 'hidden'
                V_box_KR.layout.visibility = 'hidden'
                
            elif self.method.value == 2:
                V_box_DL.layout.visibility = 'hidden'
                V_box_RF.layout.visibility = 'hidden'
                V_box_KR.layout.visibility = 'visible'
                
            elif self.method.value == 3:
                V_box_DL.layout.visibility = 'hidden'
                V_box_RF.layout.visibility = 'visible'
                V_box_KR.layout.visibility = 'hidden'
                
        self.method.observe(on_selection_button)
        
        return widgets.HBox([Method_Box, V_box_KR ,V_box_DL], layout={'height': '600px', 'width': '1200px'})

    def generate_materials(self):
        style = {'description_width': 'initial'}
        self.CationCell =widgets.SelectMultiple(
                options=['BPY', 'PY'],
                value=['BPY'],
                description='Cations',
                disabled=False
            )
        self.AnionCell =widgets.SelectMultiple(
                options=['PF6', 'TF', 'TFSI', 'DCA'],
                value=['TF'],
                description='Anions',
                disabled=False
            )
           
        items_layout = Layout( width='50%')     # override the default width of the button to 'auto' to let the button grow

        box_layout = widgets.Layout(display='flex',
                                flex_flow='row',
                                align_items='stretch',
                                border='None',
                                width='100%')
        self.button = widgets.Button(description="Click After Material Selection!", style=style, layout={'width': '260px'})
        output = widgets.Output()

        def on_button_clicked(b):
            output.clear_output()
            self.Materials = []
            self.dict_mat = {}
            with output:
                output_str = "Cations: "
                self.Materials = [v for v in self.CationCell.value]
                self.Materials.extend([v for v in self.AnionCell.value])                    
                for v in self.CationCell.value:
                    output_str += v+", "
                    self.dict_mat[v]= '+'
                print(output_str[:-2])
                output_str = "Anions: "
                for v in self.AnionCell.value:
                    output_str += v+", "
                    self.dict_mat[v]= '-'
                print(output_str[:-2])
            t = list(self.tab.children)
            t[2] = self.generate_design()
            self.tab.children = t
            
        self.button.on_click(on_button_clicked)
            
        self.Materials = [v for v in self.CationCell.value]
        self.Materials.extend([v for v in self.AnionCell.value])

        items = [self.CationCell, self.AnionCell, self.button, output]

        box = widgets.Box(children=items, layout=box_layout)
            
        return widgets.VBox([box], layout={'height': '600px'})
        
    def generate_design(self):
        """
        Design Tab controller
        """
        " List of possible methods: ‘Nelder-Mead’, ‘Powell’, ‘CG’, ‘BFGS’,‘Newton-CG’,‘L-BFGS-B’,‘TNC’,‘COBYLA’,‘SLSQP’,‘trust-constr’,‘dogleg’,‘trust-ncg’,‘trust-exact’,‘trust-krylov’ "
        style = {'description_width': '150px'}
        
        self.Optimization_method = widgets.Dropdown(
            options=[('Nelder-Mead', 1), ('L-BFGS-B', 2)],
            value=1,
            description='Optimization method:', style=style)
        
        self.Sigma_Steps = widgets.BoundedIntText(value=100,step=1,min=10, max = 1e7,
            description='Constraint Iterations:',
            disabled=False, continuous_update=False,
            orientation='horizontal',readout=True,
            readout_format='d', style=style)
        
        self.Min_Sigma = widgets.BoundedFloatText(value=0.01,min=0.01, max = 1e7,step=0.05,
            description='Min constraint coeff.:',
            disabled=False, continuous_update=False,
            orientation='horizontal',readout=True,
            readout_format='d', style=style)
        
        self.Max_Sigma = widgets.BoundedFloatText(value=20000,min=0., max = 1e9,step=10.0,
            description='Max constraint coeff.:',
            disabled=False, continuous_update=False,
            orientation='horizontal',readout=True,
            readout_format='d', style=style)
        
        self.progress_design = widgets.FloatProgress(
                    value=0.0,
                    min=0,
                    max=10.0,
                    step=0.1,
                    description='Progress:',
                    bar_style='info',
                    orientation='horizontal',visibility = 'hidden', style=style, layout={'width': '400px'})
        
        self.progress_design.layout.visibility = 'hidden'
        
        
        self.Optimization_submit = widgets.Button(description="Submit", style =style )
        self.output =  widgets.Output()
        
        self.savesolvent_button = widgets.Button(description="Save Result")
        self.savesolvent_name = widgets.Text(value='Result',
                                        placeholder='Type something',
                                        description='File name:',
                                        disabled=False)
                    
        self.savesolvent_button.layout.visibility = 'hidden'
        self.savesolvent_name.layout.visibility = 'hidden'
        def submit_call_back(b):
            self.progress_design.value = 0    
            self.progress_design.layout.visibility = 'visible'
            
            
            optimal_pt = np.random.uniform(0,1, len(Materials))
            for cnt, sigma in enumerate(np.logspace(self.Min_Sigma.value, self.Max_Sigma.value, self.Sigma_Steps.value )):
                optimal_pt = self.ml.minimize_func(optimal=optimal_pt, sig=sigma, i = cnt)
                k = (10*(cnt+1)/self.Sigma_Steps.value)
                self.progress_design.value = k
            self.output.clear_output()
            with self.output:
                print("Optimized Solvent: ")
                for i in range(len(Materials)):
                    if Materials[i][:-1] in self.dict_mat.keys():
                        print(Materials[i], " : ",np.round(optimal_pt[i],3))
            self.savesolvent_button.layout.visibility = 'visible'
            self.savesolvent_name.layout.visibility = 'visible'
            
            def save_fig_on_button_clicked(c):
                with open(self.savesolvent_name.value+'.dat', 'w') as f:
                    for i in range(len(Materials)):
                        if Materials[i][:-1] in self.dict_mat.keys():
                            f.write(Materials[i] +" : " +str(np.round(optimal_pt[i],3)))
                            f.write("\n")
                
            self.savesolvent_button.on_click(save_fig_on_button_clicked)
            self.savesolvent_button
            
        self.Optimization_submit.on_click(submit_call_back)
        
        Optimizer_Box = widgets.VBox([self.Optimization_method,self.Sigma_Steps, self.Min_Sigma, self.Max_Sigma, self.Optimization_submit, self.progress_design, self.output, self.savesolvent_button,  self.savesolvent_name])
        
        Warning_ouput = widgets.Output()
        
        with Warning_ouput:
            print("Cations or anions molar fractions sum should be equal to one otherwise their ratio is going to be used!")
        self.Prediction_Box = ()
        
        
        for cnt, name in enumerate(self.CationCell.value):
            if cnt == 0 :
                self.Prediction_Box += (widgets.BoundedFloatText(value=1.0,min=0,max=1.0,step=0.001,description=name,disabled=False, style=style), )
            else:
                self.Prediction_Box += (widgets.BoundedFloatText(value=0.0,min=0,max=1.0,step=0.001,description=name,disabled=False, style=style), )    

        for cnt, name in enumerate(self.AnionCell.value):
            if cnt == 0 :
                self.Prediction_Box += (widgets.BoundedFloatText(value=1.0,min=0,max=1.0,step=0.001,description=name,disabled=False, style=style), )
            else:
                self.Prediction_Box += (widgets.BoundedFloatText(value=0.0,min=0,max=1.0,step=0.001,description=name,disabled=False, style=style), )    
        self.Prediction_Box += (widgets.Button(description="Run", style=style, layout={'width': 'max-content'}), )
        
        self.Value_box = widgets.Output()
        
        HBoxPrediction = widgets.HBox([widgets.VBox(self.Prediction_Box), widgets.VBox([self.Value_box])])
        
        
        def click_button(b):
            self.Value_box.clear_output()
            with self.Value_box:
                x = np.zeros((len(Materials)))
                
                sum_molar_cat =  np.sum([self.Prediction_Box[cnt].value for cnt in range(len(self.CationCell.value))])
                
                for cnt in range(len(self.CationCell.value)):
                    print(" Molar frac. : ", np.round(self.Prediction_Box[cnt].value/sum_molar_cat,3), '\n')
                
                sum_molar_an =  np.sum([self.Prediction_Box[len(self.CationCell.value)+cnt].value for cnt in range(len(self.AnionCell.value))])
                
                for cnt in range(len(self.AnionCell.value)):
                    print(" Molar frac. : ", np.round(self.Prediction_Box[cnt+len(self.CationCell.value)].value/sum_molar_an,3),'\n')
                    
                for i in range(len(Materials)):
                    for cnt in range(len(self.CationCell.value)):
                        if Materials[i][:-1] == self.CationCell.value[cnt]:
                            x[i] = np.round(self.Prediction_Box[cnt].value/sum_molar_cat,3)
                    for cnt in range(len(self.AnionCell.value)):
                        if Materials[i][:-1] ==self.AnionCell.value[cnt]:
                             x[i] = np.round(self.Prediction_Box[cnt+len(self.CationCell.value)].value/sum_molar_an,3)
                print(" Predicted PMF : ", self.ml.kr_func(x)[0][0])    
                
        self.Prediction_Box[-1].on_click(click_button)

        accordion = widgets.Accordion(children=[Optimizer_Box, HBoxPrediction, Warning_ouput],layout={'height': '375px', 'width': '1500px'} )
        accordion.set_title(0, 'Optimization of solvent performance')
        accordion.set_title(1, 'Prediction of solvent performance  ')
        accordion.set_title(2, 'Warning for Prediction of solvent performance  ')    
        
        return widgets.VBox([accordion], layout={'height': '600px'})
        
    def generate_child(self, name):
        """
        generic child addition
        """
        return widgets.VBox([widgets.Text(description=name)], layout={'height': '600px'})
