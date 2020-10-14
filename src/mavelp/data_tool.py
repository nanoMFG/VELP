import os, sys
import numpy as np

global All_material

def read_data_global(path='./data/data.dat'):
    
    cnt = 0
    print(path)
    data = []
    with open(path, 'r') as f:
        #file = f.read().split('\n')

        All_material = f.readline().split('\t')[:-1]

        for line in f:
            data.append([float(a) for a in  line.split('\t')])
    
    data = np.array(data)
        
    return All_material, data[:,:-1], data[:,-1].reshape((-1,1))
    
