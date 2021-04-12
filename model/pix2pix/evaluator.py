import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .generator import Generator
from .discriminator import Discriminator

class Evaluator:
    def __init__(self, config, util):
        self.config = config
        self.util = util

        self.generator = Generator(config, in_channels=4)
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=self.config.gen_lr, betas=(0.5, 0.999))

        self.util.load_checkpoint(self.config.gen_path, self.generator, self.opt_gen, self.config.gen_lr)
        self.generator.eval()

    def predict(self, X):
        return self.generator(X)