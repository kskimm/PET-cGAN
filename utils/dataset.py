import os
import glob
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.preprocess import *
from utils.dicom import read_dicom


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')


def create_loader(inp_path, out_path, batch_size, shuffle = True):
    # Detect the type of input folders
    dataset = Custom_dataset()
    inp_dir = Path(inp_path)
    max_val_list = []
    for pat in list(inp_dir.iterdir()):
        temp_list = []
        arr_list = []
        for dcm_dir in pat.glob('*.*'):
            dcm = read_dicom(dcm_dir)
            arr = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope
            max_val = np.amax(arr)
            temp_list.append(max_val)
            arr_list.append(arr)
        pat_max_val = np.amax(np.array(temp_list))
        max_val_list.append(pat_max_val)
        
        for arr in arr_list:
            arr /= pat_max_val
            dataset.inp_list.append(dcm_preprocess3(arr))
    
    out_dir = Path(out_path)
    for j, pat in enumerate(list(out_dir.iterdir())):
        norm = max_val_list[j]
        for dcm_dir in pat.glob('*.*'):
            dcm = read_dicom(dcm_dir)
            arr = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope / norm
            dataset.out_list.append(dcm_preprocess3(arr))
            
    dataset.norm_list = max_val_list
        

    data_loader = DataLoader(
        dataset = dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        num_workers = 4,
        pin_memory = True
    )

    return data_loader

def create_loader_for_generate(inp_path, batch_size):
    dataset = Single_dataset_with_dcm()
    inp_dir = Path(inp_path)
    max_val_list = []
    for pat in list(inp_dir.iterdir()):
        temp_list = []
        arr_list = []
        for dcm_dir in pat.glob('*.*'):
            dataset.dir_list.append(dcm_dir)
            dcm = read_dicom(dcm_dir)
            arr = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope
            max_val = np.amax(arr)
            temp_list.append(max_val)
            arr_list.append(arr)
        pat_max_val = np.amax(np.array(temp_list))
        max_val_list.append(pat_max_val)
        
        for arr in arr_list:
            arr /= pat_max_val
            dataset.inp_list.append(dcm_preprocess3(arr))

    dataset.norm_list = max_val_list
    
    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False)
    return data_loader


class Custom_dataset(Dataset):
    def __init__(self):
        self.inp_list = []
        self.out_list = []

    def __len__(self): 
        return len(self.inp_list)

    def __getitem__(self, idx: int):
        x = self.inp_list[idx]
        y = self.out_list[idx]
        return x, y

    def num_channels(self):
        in_channel = self.inp_list[0].size()[0]
        out_channel = self.out_list[0].size()[0]
        return in_channel, out_channel 

    def data_size(self):
        size = self.inp_list[0].size()[1:]
        return size 
    

class Single_dataset_with_dcm(Custom_dataset):
    def __init__(self):
        super(Single_dataset_with_dcm, self).__init__()
        self.dir_list = [] 

    def __getitem__(self, idx: int):
        return self.inp_list[idx]

    def num_channels(self):
        return self.inp_list[0].size()[0]