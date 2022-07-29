import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from models.generator import UNet_generator
from utils.dataset import *
from utils.dicom import *
from utils.write import write_info
from Val import val

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')


def generate(option):
    inp_folder = option.inp_folder
    out_folder = option.out_folder
    weight = option.weight
    project_name = option.project_name
    #rescale = option.rescale
    
    checkpoint = torch.load(weight)
    batch_size = checkpoint['batch_size']
    
    loader_creator = create_loader_for_generate
    
    # Pre-process and get data loader.
    data_loader = loader_creator(inp_path = inp_folder, batch_size = batch_size)
    in_channels = data_loader.dataset.num_channels()
    
    fid_loader_creator = create_loader
    if out_folder is not None:  
        fid_data_loader = fid_loader_creator(inp_path = inp_folder, out_path = out_folder, batch_size = batch_size)
    
    # Allocate generator and discriminator
    g_norm_layer = nn.InstanceNorm2d if batch_size == 1 else nn.BatchNorm2d
    generator = UNet_generator(
            in_channels = in_channels, 
            out_channels = in_channels,
            use_dropout = True,
            numFold = 7, 
            norm_layer = g_norm_layer
    ).to(device)
    
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # generate the directory where the result is supposed to be saved
    results_path = ROOT / 'runs' / Path(__file__).stem / project_name
    count = 1
    project_root_name = project_name
    while results_path.exists():
        count += 1
        project_name = f'{project_root_name}{count}'
        results_path = ROOT / 'runs' / Path(__file__).stem / project_name
    
    results_path.mkdir(parents = True, exist_ok = True)
    write_info(path = results_path, option = option)
    
    img_path = results_path / 'img'
    img_path.mkdir(exist_ok = True)
    
    generate_func = generate_imgs
    
    generate_func(data_loader, generator, img_path)

    with torch.no_grad():
        if out_folder is not None:
            val(fid_data_loader, results_path, generator)
    
def generate_imgs(data_loader, generator, img_path):
    k = 0
    with torch.no_grad():
        norm_list = data_loader.dataset.norm_list
        for cond_imgs in data_loader:
            gen_imgs = generator(cond_imgs.to(device)).to(cpu).detach().numpy()
            gen_imgs = (gen_imgs + 1) / 2
            
            for i in range(len(gen_imgs)):
                img_dir = data_loader.dataset.dir_list[k]
                img_name = Path(img_dir).name
                patient_name = Path(img_dir).parent.name
                patient_path = img_path/patient_name
                if not patient_path.exists():
                    patient_path.mkdir(exist_ok=True)
                dcm = read_dicom(img_dir)
                rescale_slope = dcm.RescaleSlope
                
                gen_img = gen_imgs[i]
                pat_idx = k // 389
                norm = norm_list[pat_idx]
                gen_img *= norm / rescale_slope
                gen_img = gen_img.astype(np.uint16)
                new_img = write_dicom(gen_img, dcm)
                new_img.save_as(patient_path/ img_name)
                k += 1 

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-folder', type = str, help = 'input image path.')
    parser.add_argument('--out-folder', type = str, default = None, help = 'output image path for FID score.')
    parser.add_argument('--weight', type = str, default = None, help = 'Path of pre-trained weights of.')
    parser.add_argument('--project-name', type = str, default = 'exp')
    parser.add_argument('--rescale', action = 'store_true')
    return parser.parse_args()
    
    
if __name__ == "__main__":
    option = parse_option()
    generate(option)
    
