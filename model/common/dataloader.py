import torch
import torchvision.transforms as transforms
from .mask_transforms import RandomBorderMaskTransform, ApplyMaskTransform, MaskedAreaTransform
from .image_dataset import ImageDataset, load_filepaths

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

    entropy_ordered_filepaths = load_filepaths(config.entropy_fp, config.fp_prefix)
    train_dataset = ImageDataset(
        filepaths=entropy_ordered_filepaths, shuffle=config.shuffle, lim=config.data_lim,
        both=both, xonly=xonly, yonly=yonly
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=config.batch_size,
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory
    )
    
    return train_loader

def get_val_loader(config, num_samples=1):
    both, xonly, yonly = get_transforms(config)

    val_dataset = ImageDataset(
        directory=config.val_fp, 
        shuffle=True, lim=num_samples,
        both=both, xonly=xonly, yonly=yonly
    )

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=num_samples, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory
    )
    
    return val_loader