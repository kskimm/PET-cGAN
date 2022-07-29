from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import create_loader
from utils.metrics import *
from models.generator import UNet_generator

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')

tensor1 = torch.rand((4, 1, 12, 12))

tensor2 = torch.rand((4, 1, 12, 12))


def val(
    data_loader: DataLoader,
    project_folder: Path,
    generator: nn.Module
        ):
    
    generator.eval()
    
    FID = FID_score().to(device)
    fid_score = FID(data_loader, generator)
    psnr = PSNR()
    mean_psnr, psnrs = psnr(data_loader, generator)
    
    if project_folder is None:
        print(f'FID Score: {fid_score: .4f}')
        print(f'Average PSNR: {mean_psnr: .4f}')
    else:    
        txt_file_name = f'FID_{fid_score: .4f}.txt'
        fid_txt = open(project_folder / txt_file_name , 'w')
        print(f'FID Score: {fid_score: .4f}', file = fid_txt)
        print(f'Average PSNR: {mean_psnr: .4f}', file = fid_txt)
    
    # ssim = SSIM()
    # mean_ssim, ssims = ssim(data_loader, generator)
    # print(f'Average SSIM: {mean_ssim: .4f}', file = fid_txt)

def val2(
    data_loader: DataLoader,
    project_folder: Path
        ):
    
    FID = FID_score().to(device)
    fid_score = FID.dataset_compare(data_loader)
    psnr = PSNR()
    mean_psnr, psnrs = psnr.dataset_compare(data_loader)
    
    if project_folder is None:
        print(f'FID Score: {fid_score: .4f}')
        print(f'Average PSNR: {mean_psnr: .4f}')
    else:    
        txt_file_name = f'FID_{fid_score: .4f}.txt'
        fid_txt = open(project_folder / txt_file_name , 'w')
        print(f'FID Score: {fid_score: .4f}', file = fid_txt)
        print(f'Average PSNR: {mean_psnr: .4f}', file = fid_txt)
    
    

def run(
    inp_folder,
    out_folder
):

    # Pre-process and get data loader.
    data_loader = create_loader(inp_path = inp_folder, out_path = out_folder, batch_size = 4)
    val2(data_loader, None)
    
    
if __name__ == '__main__':
    folders = ['FBP', 'FBP_by3D', 'OSEM2D', 'OSEM2D_by3D', 'TrueX', 'TrueX_by3D']
    
    for folder in folders:
        print(f'\n{folder}')
        run(
            inp_folder = f'D:\Programs\Translation\\runs\generate\{folder}\img',
            out_folder = 'D:\Data\Processed\PET\OSEM3D_4\\test',
        )

    