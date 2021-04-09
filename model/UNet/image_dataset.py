import os
import torch
import random
from PIL import Image

class UNetImageDataset(torch.utils.data.Dataset):

    def __init__(self, directory=None, both=None, xonly=None, yonly=None, shuffle=False, filepaths=[], lim=None):
        super(UNetImageDataset).__init__()
        self.filepaths = filepaths or list(self.get_pic_filepaths(directory))
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.filepaths)
        if lim and lim > 0:
            self.filepaths = self.filepaths[:lim]
        identity = lambda x: x
        self.transform_both = both or identity
        self.transform_xonly = xonly or identity
        self.transform_yonly = yonly or identity

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        t = self.transform_both(img)
        return self.transform_xonly(t), self.transform_yonly(t)
    
    def get_pic_filepaths(self, directory):
        for folder in os.listdir(directory):
            for pic in os.listdir(os.path.join(directory, folder)):
                yield os.path.join(directory, folder, pic)

    def load_filepaths(directory, prefix=None):
        filepaths = []
        with open(directory) as f:
            for path in f.readlines():
                if prefix is not None:
                    filepaths.append(f"{prefix}{path.strip()}")
                else:
                    filepaths.append(path.strip())
        return filepaths
