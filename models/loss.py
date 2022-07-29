import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')
MSE = nn.MSELoss(reduction = 'mean')

class Adversarial_loss(object):
    def __init__(self, criterion = nn.BCEWithLogitsLoss()):
        self.criterion = criterion

    def __call__(self, input, target):
        if target == True:
            labels = torch.FloatTensor(input.size()).fill_(1.0)
            labels.require_grad = False
        else:
            labels = torch.FloatTensor(input.size()).fill_(0.0)
            labels.require_grad = False

        return self.criterion(input, labels.to(device))
    
    
class Wasserstein_loss(object):
    def __init__(self):
        pass

    def __call__(self, input, target):
        if target == True:
            loss = -input.view(-1).mean()
        else:
            loss = input.view(-1).mean()

        return loss


class _VGG_based_loss(nn.Module):
    def __init__(self, resize, num_layer, bn):
        super(_VGG_based_loss, self).__init__()
        if bn and num_layer == 19:
            network = torchvision.models.vgg19_bn(pretrained=True)
        elif not bn and num_layer == 19:
            network = torchvision.models.vgg19(pretrained=True)
        if bn and num_layer == 16:
            network = torchvision.models.vgg16_bn(pretrained=True)
        elif not bn and num_layer == 16:
            network = torchvision.models.vgg16(pretrained=True)

        self.blocks = self.get_block(network)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def get_block(self, network):
        blocks = []
        start = 0
        for i, layer in enumerate(network.features[:]):
            if isinstance(layer, nn.MaxPool2d):
                end = i
                blocks.append(network.features[start:end])
                start = end
        
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
            
        return torch.nn.ModuleList(blocks)
    
    def preprocessing(self, input):
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)

        input = (input - self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)

        return input

    @classmethod
    def _get_content_loss_for_layer(cls, inp1, inp2):
        return MSE(inp1, inp2)
    
    # @classmethod
    # def _get_gram_matrix(cls, inp):
    #     b, c, h, w = inp.shape #batch, channel, height, width
    #     gram_map = inp.view(b*c, h*w)
    #     gram_matrix = torch.mm(gram_map, gram_map.t())
    #     return torch.div(gram_matrix, b*c*h*w)
    
    @classmethod
    def _get_gram_matrix(cls, inp):
        b, c, h, w = inp.shape #batch, channel, height, width
        gram_map = inp.view(b, c, h*w)
        gram_matrix = torch.bmm(gram_map, gram_map.transpose(dim0 = 2, dim1 = 1))
        return torch.div(gram_matrix, c*h*w)
    
    @classmethod
    def _get_style_loss_for_layer(cls, inp1, inp2):
        return MSE(cls._get_gram_matrix(inp1), cls._get_gram_matrix(inp2))
    
    def get_blockwise_map(self, input):
        input = self.preprocessing(input)
        feature_list = [] 
        for bl in self.blocks:
            input = bl(input)
            feature_list.append(input.cpu())
        del input
        return feature_list

    def get_gram_matrix(self, input):
        input = self.preprocessing(input)
        gm_list = [] 
        for bl in self.blocks:
            input = bl(input)
            gm_list.append(self._get_gram_matrix(input))
        return gm_list

    def deconv(self):
        pass

class VGG_perceptual_loss(_VGG_based_loss):
    def __init__(self, resize=True, num_layer = 19, bn = False):
        super(VGG_perceptual_loss, self).__init__(resize, num_layer, bn)
       
    def forward(self, input, target):
        input = self.preprocessing(input)
        target = self.preprocessing(target)

        style_loss = 0
        content_loss = 0

        for i, bl in enumerate(self.blocks):
            input = bl(input)
            target = bl(target)
            content_loss += self._get_content_loss_for_layer(input, target)
            style_loss += self._get_style_loss_for_layer(input, target)
        return content_loss + style_loss
    
    def test(self, input, target):
        input = self.preprocessing(input)
        target = self.preprocessing(target)

        style_loss = 0
        content_loss = 0

        for i, bl in enumerate(self.blocks):
            input = bl(input)
            target = bl(target)
            current_content_loss = self._get_content_loss_for_layer(input, target)
            current_style_loss = self._get_style_loss_for_layer(input, target)
            if i == 2:
                content_loss += current_content_loss
            if i < 4:
                style_loss += current_style_loss
            print("%d/%d: content loss: %.4f | style loss: %.4f" % (i, len(self.blocks), current_content_loss, current_style_loss))
        print("content loss: %.4f | style loss: %.4f" % (content_loss, style_loss))
        return content_loss + style_loss
    
    
