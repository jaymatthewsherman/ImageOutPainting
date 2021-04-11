from pix2pix import Trainer as Pix2PixTrainer
from args import config_from_args
from common import Util, get_train_loader, get_val_loader

if __name__ == "__main__":
    config = config_from_args()
    util = Util(config)
    
    trainer = Pix2PixTrainer(config, util)

    train_loader = get_train_loader(config)
    val_loader = lambda: get_val_loader(config)

    for epoch in range(config.epochs):
        trainer.train(train_loader, val_loader, epoch)

