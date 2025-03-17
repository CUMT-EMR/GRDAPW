import numpy as np
import torch 
import sys
sys.path.append("../")
import scipy
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
# import utils_BayTIDE as utils
from scipy.io import loadmat# impory solar.txt as np.array
# tensor = np.loadtxt('./data/raw/solar.txt', delimiter=',')
tensor = pd.read_excel("Outlierdatasets.xlsx").to_numpy()


tensor = tensor.T
print(tensor.shape)

print(np.isnan(tensor).any())
print(np.nanmax(tensor))
print(np.nanmin(tensor))
print(np.nanmean(tensor))
sub_tensor = tensor

data_save = {}
data_save['ndims'] = sub_tensor.shape

data_save['raw_data'] = sub_tensor 

data_save['data'] = []

data_save['time_uni'] = np.linspace(0,1,sub_tensor.shape[1])

def generate_random_mask( shape, drop_rate=0.2, valid_rate=0.1):
    """
    train_ratio: 1-valid_rate-drop_rate
    test_ratio: drop_rate
    valid_ratio: valid_rate
    """
    N,T = shape

    mask_train_list = []
    mask_test_list = []
    mask_valid_list = []

    for t in range(T):

        mask = np.random.rand(N)
        mask_train = np.where(mask>drop_rate+valid_rate, 1, 0)
        mask_test = np.where(mask<drop_rate, 1, 0)
        mask_valid = np.where((mask>drop_rate) & (mask<drop_rate+valid_rate), 1, 0)

        mask_train_list.append(mask_train)
        mask_test_list.append(mask_test)
        mask_valid_list.append(mask_valid)
    
    mask_train = np.stack(mask_train_list, axis=1)
    mask_test = np.stack(mask_test_list, axis=1)
    mask_valid = np.stack(mask_valid_list, axis=1)
    
    print(mask_train.shape)
    return mask_train, mask_test, mask_valid

fold = 5
drop_rate = 0.2
valid_rate = 0.1

for i in range(fold):
    mask_train, mask_test, mask_valid = generate_random_mask(sub_tensor.shape, drop_rate, valid_rate)
    data_save['data'].append({'mask_train':mask_train, 'mask_test':mask_test, 'mask_valid':mask_valid})

file_name = '../solar_impute_guiyi'+'_r_%.1f'%(drop_rate)+'.npy'
np.save(file_name, data_save)
mask_train.sum()/mask_train.size

