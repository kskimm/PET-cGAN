import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from models.generator import UNet_generator
from utils.dataset import create_loader_for_generate, create_loader
from utils.plots import gen_plot

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')


def generate(option):
    inp_folder = option.inp_folder
    weight = option.weight
    
    checkpoint = torch.load(weight)
    batch_size = checkpoint['batch_size']
    
    # Pre-process and get data loader.
    data_loader = create_loader_for_generate(inp_path = inp_folder, batch_size = batch_size)
    in_channels = data_loader.dataset.num_channels()
    
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
    
    gen_plot(val_data_loader = data_loader, generator = generator)
    


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-folder', type = str, help = 'input image path.')
    parser.add_argument('--weight', type = str, default = None, help = 'Path of pre-trained weights of.')
    return parser.parse_args()
    
    
if __name__ == "__main__":
    option = parse_option()
    generate(option)
    
