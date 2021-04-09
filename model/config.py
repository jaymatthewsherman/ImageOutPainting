import torch
from PIL import ImageColor

# Hyperparameters

GEN_LEARNING_RATE = 2e-4
DISC_LEARNING_RATE = 2e-4
L1_LAMBDA = 100

UNET_LEARNING_RATE = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 12
NUM_WORKERS = 2
PIN_MEMORY = True
DATA_LIM = 36_000
SHUFFLE = False

ENTROPY_FILEPATHS = "dataset/entropy-filepaths-ordered.txt"
FP_PREFIX = "/dataset/"

DEFAULT_EPOCHS = 100
DEFAULT_STRIPE_WIDTH = 12
DEFAULT_PIC_DIM = (3, 256, 256)
DEFAULT_COLOR = "#000000"
DEFAULT_COLLAPSE = False
DEFAULT_OUTSIDE = True

LOAD_MODEL = True
BREAK_ON_ERROR = True
LOG_BATCH_INTERVAL = 100

import os
print(os.getcwd())

GEN_CHECKPOINT_PATH = f'{os.getcwd()}\\Pix2Pix\\saved\\pix2pix-gen-nc.checkpoint.pth.tar'
DISC_CHECKPOINT_PATH = f'{os.getcwd()}\\Pix2Pix\\saved\\pix2pix-disc-nc.checkpoint.pth.tar'

class Config:
    def __init__(self, epochs=DEFAULT_EPOCHS, 
                 stripe_width=DEFAULT_STRIPE_WIDTH, 
                 pic_dim=DEFAULT_PIC_DIM,
                 color=DEFAULT_COLOR,
                 should_collapse=DEFAULT_COLLAPSE,
                 outside=DEFAULT_OUTSIDE,
                 unet_lr = UNET_LEARNING_RATE,
                 gen_lr = GEN_LEARNING_RATE,
                 disc_lr = DISC_LEARNING_RATE):
        self.epochs = epochs
        self.stripe_width = stripe_width
        self.pic_dim = list(pic_dim)
        self.pic_channels = self.pic_dim[0]
        self.pic_height = self.pic_dim[1]
        self.pic_width = self.pic_dim[2]
        self.color = [c / 256 for c in ImageColor.getcolor(color, "RGB")]
        self.should_collapse = should_collapse
        self.outside = outside
        self.unet_lr = unet_lr,
        self.gen_lr = gen_lr,
        self.disc_lr = disc_lr

config = Config()