# MA-VELP
Machine-learning Assisted Virtual Exfoliation via Liquid Phase

Python codes are located in the src directory:
  1. data_tool.py : manages file reading and data management
  2. gui.py :  graphical user interface supports user interface and communication between various codes
  3. kernel_method.py : kernele ridge regression is implemented for MAVELP dataset and used for fitting, prediction and score, additionally it supports optimization 
  4. neural_network.py: neural network based regression is implemented to train NN for MAVELP dataset, addtionally it supports optimization of solvent composition

Data are located in the data directory:
  1. data.dat : data for MA-VELP
  
 
Usage: 
  Simply copy src and data files to the working directory, you can then use MA-VELP
  MA-VELP-App.ipynb to have access to the GUI
  
  
Future Release Note: 
  - We might move optimization to a separate code
  - Pip installation seems like a good option
  - Python test 

