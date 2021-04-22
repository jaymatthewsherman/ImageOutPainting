from pix2pix import Trainer as Pix2PixTrainer
from unet import Trainer as UNetTrainer
from args import config_from_args
from common import Util, get_train_loader, get_val_loader

if __name__ == "__main__":
    config = config_from_args()
    util = Util(config)
    
    if config.type not in ["pix2pix", "unet"]:
        exit(-1, "Must provide a supported model")
    elif config.type == "pix2pix":
        trainer = Pix2PixTrainer(config, util)
    elif config.type == "unet":
        trainer = UNetTrainer(config, util)
        config.should_collapse = True

    train_loader = get_train_loader(config)
    val_loader = lambda num_samples: get_val_loader(config, num_samples)

    for epoch in range(config.epochs):
        trainer.train(train_loader, val_loader, epoch)

