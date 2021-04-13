import torch
from abc import ABC, abstractmethod
from random import randrange

# Abstract class to extend for different types of masks
class MaskGenerator(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def generate(self):
        pass

# Gets a mask matrix with the mask applied on the right side
class RightStripeMaskGenerator(MaskGenerator):
    def __init__(self, config):
        super()
        self.zeros_shape = (config.pic_height, config.pic_width - config.stripe_width)
        self.ones_shape = (config.pic_height, config.stripe_width)
    
    def generate(self):
        return torch.hstack((
            torch.zeros(size=self.zeros_shape, dtype=torch.bool),
            torch.ones(size=self.ones_shape, dtype=torch.bool)
        ))
    
# Gets a mask matrix with the mask applied on the top side
class TopStripeMaskGenerator(MaskGenerator):
    def __init__(self, config):
        super()
        self.zeros_shape = (config.pic_height - config.stripe_width, config.pic_width)
        self.ones_shape = (config.stripe_width, config.pic_width)
    
    def generate(self):
        return torch.vstack((
            torch.ones(size=self.ones_shape, dtype=torch.bool),
            torch.zeros(size=self.zeros_shape, dtype=torch.bool)
        ))

# Gets a mask matrix with the mask applied randomly to any of the four sides
class RandomBorderStripeMaskGenerator():
    def __init__(self, config):
        super()
        zeros_shape = (config.pic_height - config.stripe_width, config.pic_width)
        ones_shape = (config.stripe_width, config.pic_width)
        self.top_stripe_mask = torch.vstack((
            torch.ones(size=ones_shape, dtype=torch.bool),
            torch.zeros(size=zeros_shape, dtype=torch.bool)
        ))

    def generate(self):
        return torch.rot90(self.top_stripe_mask, randrange(4))

# Applies a mask to an image
class MaskApplier():
    def __init__(self, config, in_place=False, extra_dim=True):
        self.config = config
        self.in_place = in_place
        self.extra_dim = extra_dim
        
    def apply(self, mask, img):        
        if not self.in_place:
            img = img.detach().clone()
        if img.shape[0] != self.config.pic_channels:
            img = self.fix_img(img)
        for dim in range(self.config.pic_channels):
            img[dim][mask] = self.config.color[dim]
        if self.extra_dim:
            return torch.cat((img, mask.repeat(1, 1, 1)))
        else:
            return img

    def fix_img(self, img):
        if img.shape[0] < self.config.pic_channels:
            img = img[0, :].repeat(self.config.pic_channels, 1, 1)
        else:
            img = img[:self.config.pic_channels, :]
        return img