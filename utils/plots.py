import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.utils as utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def img_plot(img, title = None, seconds = 0): 
    img = utils.make_grid(img.cpu().detach()).permute(1, 2, 0)
    img = (img+1)/2
    plt.imshow(img.numpy()) 

    plt.axis('off')

    if title is not None:
        plt.title(title)
                
    if seconds == 0:
        plt.show()
    else:
        plt.show(block = False)
        plt.pause(seconds)
    plt.close()

def batch_plot(val_data_loader, generator):
    cond_img, real_img = next(iter(val_data_loader))
    gen_img = generator(cond_img.to(device)).cpu()
    img_list = [cond_img, gen_img, real_img]
    for img in img_list:
        img = img.repeat(1, 3, 1, 1)    
    plot_tensor = torch.cat(img_list, dim = 2)
    img_plot(img = plot_tensor, seconds = 0)
    
def plot_with_idx(data_loader, generator, idx):
    cond_img = torch.unsqueeze(data_loader.dataset.inp_list[idx], 0)
    real_img = torch.unsqueeze(data_loader.dataset.out_list[idx], 0)
    gen_img = generator(cond_img.to(device)).cpu()
    img_list = [cond_img, gen_img, real_img]
    for img in img_list:
        img = img.repeat(1, 3, 1, 1)    
    plot_tensor = torch.cat(img_list, dim = 3)
    img_plot(img = plot_tensor, seconds = 0)
    

def gen_plot(val_data_loader, generator):
    cond_img = next(iter(val_data_loader))
    gen_img = generator(cond_img.to(device)).cpu()
    img_list = [cond_img, gen_img]
    for img in img_list:
        img = img.repeat(1, 3, 1, 1)    
    plot_tensor = torch.cat(img_list, dim = 2)
    img_plot(img = plot_tensor, seconds = 0)

def loader_plot(data_loader):
    cond_img, real_img = next(iter(data_loader))
    img_list = [cond_img, real_img]
    for img in img_list:
        img = img.repeat(1, 3, 1, 1)    
    plot_tensor = torch.cat(img_list, dim = 2)
    img_plot(img = plot_tensor, seconds = 0)