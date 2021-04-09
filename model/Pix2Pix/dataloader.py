import sys
sys.path.insert(0, '..')
from config import *
import torch
import torchvision.transforms as transforms
from model.masking.mask_transforms import RandomBorderMaskTransform, MaskedAreaTransform, ApplyMaskTransform
from Pix2Pix.image_dataset import Pix2PixImageDataset

def get_transforms(config):
    both_transform = transforms.Compose([
        transforms.Resize((config.pic_height, config.pic_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        RandomBorderMaskTransform(config)
    ])

    xonly_transform = ApplyMaskTransform(config, extra_dim=(not config.should_collapse))
    yonly_transform = MaskedAreaTransform(config)

    return both_transform, xonly_transform, yonly_transform


def get_train_loader(config):
    both, xonly, yonly = get_transforms(config)

    entropy_ordered_filepaths = Pix2PixImageDataset.load_filepaths(ENTROPY_FILEPATHS, prefix=FP_PREFIX)
    entropy_ordered_filepaths = ["/".join(eof.split("/")[1:]) for eof in entropy_ordered_filepaths]
    train_dataset = Pix2PixImageDataset(filepaths=entropy_ordered_filepaths, shuffle=SHUFFLE, lim=DATA_LIM,
        both=both, xonly=xonly, yonly=yonly)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    return train_loader

def get_val_loader(config, num_samples=1):
    both, xonly, yonly = get_transforms(config)

    val_dataset = Pix2PixImageDataset(directory='../places365_standard/val', shuffle=True,
        both=both, xonly=xonly, yonly=yonly, lim=num_samples)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    return val_loader
