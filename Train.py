import argparse
import os
import sys
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from models.generator import UNet_generator, Resnet_generator
from models.discriminator import Patch_discriminator
from models.common import weights_init_normal
from models.update import *
from models.loss import *
from utils.dataset import create_loader
from utils.metrics import FID_score
from utils.write import write_info

FILE = Path(__file__).resolve()
print(FILE)
ROOT = FILE.parents[0]
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')


def train(option):
    inp_folder = option.inp_folder
    out_folder = option.out_folder
    val_inp_folder = option.val_inp_folder
    val_out_folder = option.val_out_folder
    adv_loss = option.adv_loss
    add_loss = option.add_loss
    add_loss_weight = option.add_loss_weight
    epochs = option.epochs
    batch_size = option.batch_size
    optimizer = option.optimizer
    project_name = option.project_name
    lr = option.learning_rate
    #rescale = option.rescale
    # Pre-process and get data loader.

   
    loader_creator = create_loader
        
    data_loader = loader_creator(inp_path = inp_folder, out_path = out_folder, batch_size = batch_size)
    
    if val_inp_folder is not None and val_out_folder is not None:
        val_data_loader = loader_creator(inp_path = val_inp_folder, out_path = val_out_folder, batch_size = batch_size)
    else:
        val_data_loader = data_loader
    in_channels, out_channels = data_loader.dataset.num_channels()
    
    # Allocate generator and discriminator
    g_norm_layer = nn.InstanceNorm2d if batch_size == 1 else nn.BatchNorm2d
    # generator = UNet_generator(
    #         in_channels = in_channels, 
    #         out_channels = out_channels,
    #         use_dropout = True,
    #         numFold = 7, 
    #         norm_layer = g_norm_layer
    # ).to(device)
    
    generator = Resnet_generator(
            in_channels = in_channels, 
            out_channels = out_channels,
            use_dropout = False,
            norm_layer = g_norm_layer
    ).to(device)
    
    generator.apply(weights_init_normal)

    d_norm_layer = nn.Identity if adv_loss == 'WGAN' else nn.BatchNorm2d
    discriminator = Patch_discriminator(
            cond_channels = in_channels, 
            real_channels = out_channels,
            norm_layer = d_norm_layer
    ).to(device)
    discriminator.apply(weights_init_normal)
    
    betas = (0.5, 0.999)
    
    if optimizer == 'SGD':
        gen_optimizer = optim.SGD(generator.parameters(), lr, betas)
        dis_optimizer = optim.SGD(discriminator.parameters(), lr, betas)
    else:
        gen_optimizer = optim.Adam(generator.parameters(), lr, betas)
        dis_optimizer = optim.Adam(discriminator.parameters(), lr, betas)
    
    
    # Set adversarial loss
    
    if adv_loss == 'GAN':
        gan_model = GAN()

    elif adv_loss == 'WGAN':
        gan_model = WGAN()
    
    else:
        gan_model = LSGAN()
    
    # Set additional loss
    if add_loss == 'L1':
        addit_loss = nn.L1Loss()
        
    else:
        addit_loss = VGG_perceptual_loss().to(device).eval()     

    #metric
    FID = FID_score().to(device)
    
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
    
    history_file = open(results_path / 'learning history.txt', 'w')
    best_score = 10000
    best_epoch = 0

    # Learning epochs
    for epoch in range(epochs):
        #update discriminator
        generator.train()
        discriminator.train()
        
        gan_model.update(
             discriminator = discriminator, 
             dis_optimizer = dis_optimizer, 
             generator = generator, 
             gen_optimizer = gen_optimizer, 
             addit_loss = addit_loss, 
             add_loss_weight = add_loss_weight, 
             data_loader = data_loader
        )
        
        # validation
        generator.eval()
        with torch.no_grad():
            score = FID(val_data_loader, generator)
            
            if score < best_score and epoch > epochs // 2:
                best_score = copy.deepcopy(score)
                best_epoch = copy.deepcopy(epoch)
                best_gen_info = copy.deepcopy(generator.state_dict())
                best_dis_info = copy.deepcopy(discriminator.state_dict())
    
            #print(f'Epoch: {epoch} | {epochs} || FID Score: {score: .4f} || best epoch: {best_epoch} || Best FID Score: {best_score: .4f} ' )
            print(f'Epoch: {epoch} | {epochs} || FID Score: {score: .4f}', file = history_file)
    
    # save networks
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'batch_size': batch_size
    }, results_path / 'model.pth')
            
    torch.save({
        'generator': best_gen_info,
        'discriminator': best_dis_info,
        'batch_size': batch_size
    }, results_path / 'best_model.pth')


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-folder', type = str, help = 'input image path.')
    parser.add_argument('--out-folder', type = str, help = 'ground-truth image path.')
    parser.add_argument('--val-inp-folder', type = str, default = None, help = 'input image for validation path.')
    parser.add_argument('--val-out-folder', type = str, default = None, help = 'ground-truth image for validation path.')
    parser.add_argument('--adv-loss', type = str, default = 'LSGAN', choices = ['GAN', 'WGAN', 'LSGAN'], help = 'type of adversarial loss')
    parser.add_argument('--add-loss', type = str, default = 'VGG', choices = ['L1', 'VGG'], help = 'type of additional loss')
    parser.add_argument('--add-loss-weight', type = float, default = 1, help = 'weight for additional loss in updating network')
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch-size', type = int, default = 4)
    parser.add_argument('--learning-rate', type = float, default = 0.0002)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam', help='optimizer')
    parser.add_argument('--initial-weight', type = str, default = None, help = 'directory of initial network weight if you wanna fine-tune.')
    parser.add_argument('--project-name', type = str, default = 'exp')
    parser.add_argument('--rescale', action='store_true')
    return parser.parse_args()
    
    
if __name__ == "__main__":
    option = parse_option()
    train(option)
    
