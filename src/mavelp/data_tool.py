import os, sys
import numpy as np

global All_material

def read_data_global(path='./data/data.dat'):
    
    cnt = 0
    print(path)
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
    
