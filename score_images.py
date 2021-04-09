import os, torch, numpy as np, sys
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from torchvision.io import read_image
import tqdm, argparse
from random import seed, randrange, random, shuffle
import math
import matplotlib.pyplot as plt
import torch, os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.io import read_image
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
from PIL import ImageColor, Image

DEFAULT_EPOCHS = 10
DEFAULT_STRIPE_WIDTH = 24
DEFAULT_PIC_DIM = (3, 256, 256)
DEFAULT_COLOR = "#ff69b4"

# Program configuration block, should not be used for hyper parameters
# ... that would be the job of a ModelConfig block
class Config:
    def __init__(self, epochs=DEFAULT_EPOCHS, 
                 stripe_width=DEFAULT_STRIPE_WIDTH, 
                 pic_dim=DEFAULT_PIC_DIM,
                 color=DEFAULT_COLOR):
        self.epochs = epochs
        self.stripe_width = stripe_width
        self.pic_dim = list(pic_dim)
        self.pic_channels = self.pic_dim[0]
        self.pic_height = self.pic_dim[1]
        self.pic_width = self.pic_dim[2]
        self.color = [c / 256 for c in ImageColor.getcolor(color, "RGB")]
        
    def default():
        return Config()
        
config = Config.default()

def get_pic_filepaths(traindir):
    for folder in os.listdir(traindir):
        for pic in os.listdir(os.path.join(traindir, folder)):
            yield os.path.join(traindir, folder, pic)
            
mask = torch.ones((256, 256), dtype=torch.bool)
mask[config.stripe_width:config.pic_height-config.stripe_width, 
     config.stripe_width:config.pic_height-config.stripe_width] = False
disks = disk(5)

def entropy_score(img, mask):
    return sum(
        entropy(img_as_ubyte(img[dim]), disks, mask=mask)[mask].mean() 
        for dim in range(config.pic_channels))

def scored_image(filepath):
    score = entropy_score(read_image(filepath), mask)
    return score, filepath
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score some images')
    parser.add_argument('traindir', type=str, help="training directory location")
    parser.add_argument('fro', type=int, help="from percentage to operate onto")
    parser.add_argument('to', type=int, help="to percentage to operate onto")
    args = parser.parse_args()
    
    with open("entropy-filepaths.txt", "a+") as f:
        filepaths = list(get_pic_filepaths(args.traindir))
        start = int(args.fro / 100 * len(filepaths))
        end = int(args.to / 100 * len(filepaths))
        working_paths = filepaths[start:end+1]
        
        for fp in tqdm(working_paths):
            score, file = scored_image(fp)
            f.write(f"{score}\t{file}\n")
    f.close()
    

    