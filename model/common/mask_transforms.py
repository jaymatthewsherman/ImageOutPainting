import torch
from .mask_gen import TopStripeMaskGenerator, RightStripeMaskGenerator, MaskApplier, RandomBorderStripeMaskGenerator
from random import random

class Datum:
    def __init__(self, img, mask, right_side):
        self.img = img
        self.mask = mask
        self.right_side = right_side

# Transform that returns tuple of generated mask and X
# applied to both X and y so y knows about the mask
class RandomRightTransform:
    def __init__(self, config, right_chance=0.5):
        self.config = config
        self.tsmg = TopStripeMaskGenerator(config)
        self.rsmg = RightStripeMaskGenerator(config)
        self.right_chance = right_chance
    
    def __call__(self, img):
        is_right_mask = random() < self.right_chance
        mask = (self.rsmg if is_right_mask else self.tsmg).generate()
        return Datum(img, mask, is_right_mask)

# X only transform for all borders
class RandomBorderMaskTransform:
    def __init__(self, config):
        self.config = config
        self.rbsmg = RandomBorderStripeMaskGenerator(config)
    
    def __call__(self, img):
        mask = self.rbsmg.generate()
        return Datum(img, mask, None)

# X only transform to apply the given mask
class ApplyMaskTransform:
    def __init__(self, config, extra_dim=True):
        self.config = config
        self.ma = MaskApplier(config, in_place=False, extra_dim=extra_dim)
        
    def __call__(self, datum):
        return self.ma.apply(datum.mask, datum.img)

# Y only transform to get the masked out portion of the image
class MaskedAreaTransform:
    def __init__(self, config):
        self.w = config.pic_width
        self.h = config.pic_height
        self.sw = config.stripe_width
        self.c = config.pic_channels
        self.collapse = config.should_collapse
    
    def __call__(self, datum):
        # If we're producing the entire image, just return the image
        # Note that config.should_collapse is set to True for all Unet models
        if not self.collapse:
            return datum.img

        # Determine if the mask is horizontal or vertical
        if (datum.mask[self.h//2][0] ^ datum.mask[self.h//2][self.w-1]):
            return datum.img[:, datum.mask].reshape(self.c, self.h, self.sw)
        else:
            # transpose if it is vertical (either top side or bot side of image)
            return torch.transpose(datum.img[:, datum.mask].reshape(self.c, self.sw, self.w), 1, 2)
