import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.dicom import read_dicom

class DCM_to_Ndarray(object):
    def __init__(self):
        pass

    def __call__(self, file_directory):
        array = read_dicom(file_directory).pixel_array
        return array.astype(np.float32)

class DCM_to_Ndarray_scaled(object):
    def __init__(self):
        pass

    def __call__(self, file_directory):
        dcm = read_dicom(file_directory)
        array = dcm.pixel_array
        array = array.astype(np.float32) * dcm.RescaleSlope
        return array

class Naive_weighting(object):
    def __init__(self, tangent = 1./65535, arr_min = 0):
        self.tangent = tangent
        self.arr_min = arr_min

    def __call__(self, array):
        return (array - self.arr_min) * self.tangent

class Max_weighting(object):
    def __init__(self):
        pass

    def __call__(self, array):
        arr_min = np.amin(array)
        tangent = 1. / (np.amax(array) - arr_min)
        return (array - arr_min) * tangent

class Crop(object):
    def __init__(self, size, top, left):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.output_size = (size, size)
        else:
            assert len(size) == 2
            self.output_size = size

        self.top = top
        self.left = left

    def __call__(self, image):
        new_h, new_w = self.output_size
        return image[self.top: self.top + new_h, self.left: self.left + new_w]


class RandomCrop(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.output_size = (size, size)
        else:
            assert len(size) == 2
            self.output_size = size

    def __call__(self, image, set_new_random = True):
        if image.ndim == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape

        new_h, new_w = self.output_size

        if set_new_random:
            self.top = torch.randint(0, h - new_h, (1,)).item()
            self.left = torch.randint(0, w - new_w, (1,)).item()

        return image[self.top: self.top + new_h, self.left: self.left + new_w]

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if image.ndim == 3:
            image = torch.unsqueeze(image, dim = 0)
        
        image = F.interpolate(image, (self.size, self.size))
        
        if image.shape[1] != 1:
            image = torch.squeeze(image)
        else:
            image = torch.unsqueeze(torch.squeeze(image), dim = 0)
        
        return image

class CenterCrop(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.output_size = (size, size)
        else:
            assert len(size) == 2
            self.output_size = size

    def __call__(self, image):
        if image.ndim == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape

        half_size = self.output_size[0] // 2
        half_row, half_col = h // 2, w // 2

        return image[-half_size + half_row: half_size + half_row, -half_size + half_col: half_size + half_col]
    
    

dcm_preprocess = transforms.Compose([
    DCM_to_Ndarray(),
    Naive_weighting(),
    CenterCrop(size = 128),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dcm_preprocess2 = transforms.Compose([
    DCM_to_Ndarray(),
    Max_weighting(),
    CenterCrop(size = 128),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dcm_preprocess3 = transforms.Compose([
    transforms.ToTensor(),
    Resize(128),
    transforms.Normalize((0.5), (0.5))
])