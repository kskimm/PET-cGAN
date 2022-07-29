import torch
import torch.nn as nn
import torchvision
import scipy.linalg as linalg
import numpy as np
import torchmetrics.image.fid as tif

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')
sqrtm = tif.MatrixSquareRoot.apply
MSE = nn.MSELoss()
class FID_score(nn.Module):
    def __init__(self):
        super(FID_score, self).__init__()
        inception_v3 = torchvision.models.inception_v3(pretrained = True)
        self.Conv2d_1a_3x3 = inception_v3.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception_v3.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception_v3.Conv2d_2b_3x3
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = inception_v3.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception_v3.Conv2d_4a_3x3
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_v3.Mixed_5b
        self.Mixed_5c = inception_v3.Mixed_5c
        self.Mixed_5d = inception_v3.Mixed_5d
        self.Mixed_6a = inception_v3.Mixed_6a
        self.Mixed_6b = inception_v3.Mixed_6b
        self.Mixed_6c = inception_v3.Mixed_6c
        self.Mixed_6d = inception_v3.Mixed_6d
        self.Mixed_6e = inception_v3.Mixed_6e
        self.Mixed_7a = inception_v3.Mixed_7a
        self.Mixed_7b = inception_v3.Mixed_7b
        self.Mixed_7c = inception_v3.Mixed_7c
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
        blocks = [
            self.Conv2d_1a_3x3,
            self.Conv2d_2a_3x3,
            self.Conv2d_2b_3x3,
            self.maxpool1,
            self.Conv2d_3b_1x1,
            self.Conv2d_4a_3x3,
            self.maxpool2,
            self.Mixed_5b,
            self.Mixed_5c,
            self.Mixed_5d,
            self.Mixed_6a,
            self.Mixed_6b,
            self.Mixed_6c,
            self.Mixed_6d,
            self.Mixed_6e,
            self.Mixed_7a,
            self.Mixed_7b,
            self.Mixed_7c,
            self.avgpool
        ]
    
        for block in blocks:
            block.requires_grad = False

        self.layer = nn.Sequential(*blocks) 
        self.transform = nn.functional.interpolate

    def forward(self, input):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)

        input = self.transform(input, mode='bilinear', size=(299, 299), align_corners=False)
        self.layer.eval()
        x = self.layer(input)
 
        return x

    def __call__(self, data_loader, generator):
        assert isinstance(generator, nn.Module)
        eps=1e-6
        
        gen_features = []
        gt_features = []
        
        for cond_img, ground_truth in data_loader:
            cond_img, ground_truth = cond_img.to(device = device).detach(), ground_truth.to(device = device).detach()
            gen_img = generator(cond_img)
            gen_feature, gt_feature = self.forward(gen_img), self.forward(ground_truth)
            gen_feature = gen_feature.to(cpu).squeeze(dim = 2).squeeze(dim = 2).detach().numpy()
            gt_feature = gt_feature.to(cpu).squeeze(dim = 2).squeeze(dim = 2).detach().numpy()
            gen_features.append(gen_feature)
            gt_features.append(gt_feature)
    
        gen_features = np.concatenate(gen_features)
        gt_features = np.concatenate(gt_features)

        gen_mean, gen_cov = get_mu_and_sigma(gen_features)
        gt_mean, gt_cov = get_mu_and_sigma(gt_features)

        diff = gen_mean - gt_mean

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(gen_cov.dot(gt_cov), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(gen_cov.shape[0]) * eps
            covmean = linalg.sqrtm((gen_cov + offset).dot(gt_cov + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(gen_cov) + np.trace(gt_cov) - 2 * tr_covmean)

    
    def dataset_compare(self, data_loader):
        eps=1e-6
        
        gen_features = []
        gt_features = []
        
        for cond_img, ground_truth in data_loader:
            gen_img, ground_truth = cond_img.to(device = device).detach(), ground_truth.to(device = device).detach()
            gen_feature, gt_feature = self.forward(gen_img), self.forward(ground_truth)
            gen_feature = gen_feature.to(cpu).squeeze(dim = 2).squeeze(dim = 2).detach().numpy()
            gt_feature = gt_feature.to(cpu).squeeze(dim = 2).squeeze(dim = 2).detach().numpy()
            gen_features.append(gen_feature)
            gt_features.append(gt_feature)
    
        gen_features = np.concatenate(gen_features)
        gt_features = np.concatenate(gt_features)

        gen_mean, gen_cov = get_mu_and_sigma(gen_features)
        gt_mean, gt_cov = get_mu_and_sigma(gt_features)

        diff = gen_mean - gt_mean

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(gen_cov.dot(gt_cov), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(gen_cov.shape[0]) * eps
            covmean = linalg.sqrtm((gen_cov + offset).dot(gt_cov + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(gen_cov) + np.trace(gt_cov) - 2 * tr_covmean)


def get_mu_and_sigma(matrix):
    return np.mean(matrix, axis = 0), np.cov(matrix, rowvar = False)

class PSNR(object):
    def __init__(self):
        self.max_val = torch.tensor(1.)
    
    def __call__(self, data_loader, generator):
        assert isinstance(generator, nn.Module)
        
        PSNRs = []
        for cond_img, ground_truth in data_loader:
            cond_img, ground_truth = cond_img.to(device = device).detach(), ground_truth.to(device = device).detach()
            gen_imgs = generator(cond_img).detach()
            
            for gen_img, gt_img in zip(gen_imgs, ground_truth):
                gen_img = (gen_img + 1)/2.
                gt_img = (gt_img + 1)/2.
                    
                mse = MSE(gen_img, gt_img)
                PSNR_val = 20*torch.log10(self.max_val) - 10*torch.log10(mse)
                
                PSNRs.append(PSNR_val.item())
        
        return torch.mean(torch.tensor(PSNRs)).item(), PSNRs

    def dataset_compare(self, data_loader):
        PSNRs = []
    
        for cond_img, ground_truth in data_loader:
            cond_img, ground_truth = cond_img.to(device = device).detach(), ground_truth.to(device = device).detach()
            
            for gen_img, gt_img in zip(cond_img, ground_truth):
                
                gen_img = (gen_img + 1)/2.
                gt_img = (gt_img + 1)/2.
                    
                mse = MSE(gen_img, gt_img)
                PSNR_val = 20*torch.log10(self.max_val) - 10*torch.log10(mse)
                
                PSNRs.append(PSNR_val.item())
        
        return torch.mean(torch.tensor(PSNRs)).item(), PSNRs
        
        
class SSIM(object):
    def __init__(self):
        k1 = 0.01
        k2 = 0.03
        L = 1
        self.c1 = (k1*L)**2
        self.c2 = (k2*L)**2
        
    def __call__(self, data_loader, generator):
        assert isinstance(generator, nn.Module)
        
        SSIMs = []
        for cond_img, ground_truth in data_loader:
            cond_img, ground_truth = cond_img.to(device = device).detach(), ground_truth.to(device = device).detach()
            gen_imgs = generator(cond_img).detach()
            
            for gen_img, gt_img in zip(gen_imgs, ground_truth):
                gen_img = gen_img.view(-1,1)
                gt_img = gt_img.view(-1,1)
                
                gen_img = (gen_img + 1)/2.
                gt_img = (gt_img + 1)/2.
                
                gen_mean = torch.mean(gen_img).item()
                gen_std = torch.std(gen_img, unbiased = False).item()
                
                gt_mean = torch.mean(gt_img).item()
                gt_std = torch.std(gt_img, unbiased = False).item()
                mat = torch.cat((gen_img, gt_img), dim = 1)
                cov_matrix = torch.cov(torch.transpose(mat, 1, 0))
                cov = cov_matrix[0, 1]
                SSIM_val = (2*gen_mean*gt_mean + self.c1) / (gen_mean**2 + gt_mean**2 + self.c1) * (2*cov + self.c2) / (gen_std**2 + gt_std**2 + self.c2)
                SSIMs.append(SSIM_val.item())
                
        return torch.mean(torch.tensor(SSIMs)).item(), SSIMs
    
    def dataset_compare(self, data_loader):
            
        SSIMs = []
        for cond_img, ground_truth in data_loader:
            cond_img, ground_truth = cond_img.to(device = device).detach(), ground_truth.to(device = device).detach()
            
            for gen_img, gt_img in zip(cond_img, ground_truth):
                gen_img = gen_img.view(-1,1)
                gt_img = gt_img.view(-1,1)
                
                gen_img = (gen_img + 1)/2.
                gt_img = (gt_img + 1)/2.
                
                gen_mean = torch.mean(gen_img).item()
                gen_std = torch.std(gen_img, unbiased = False).item()
                
                gt_mean = torch.mean(gt_img).item()
                gt_std = torch.std(gt_img, unbiased = False).item()
                mat = torch.cat((gen_img, gt_img), dim = 1)
                cov_matrix = torch.cov(torch.transpose(mat, 1, 0))
                cov = cov_matrix[0, 1]
                SSIM_val = (2*gen_mean*gt_mean + self.c1) / (gen_mean**2 + gt_mean**2 + self.c1) * (2*cov + self.c2) / (gen_std**2 + gt_std**2 + self.c2)
                SSIMs.append(SSIM_val.item())
        
        return torch.mean(torch.tensor(SSIMs)).item(), SSIMs