import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .generator import Generator
from .discriminator import Discriminator
import math
import wandb
import os

class Trainer:
    def __init__(self, config, util):
        self.config = config
        self.util = util

        self.generator = Generator(config, in_channels=4).to(config.device)
        self.discriminator = Discriminator(config, in_channels=7).to(config.device)

        self.opt_gen = optim.Adam(self.generator.parameters(), lr=self.config.gen_lr, betas=(0.5, 0.999))
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=self.config.disc_lr, betas=(0.5, 0.999))

        self.bce = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        self.gen_scaler = torch.cuda.amp.GradScaler()
        self.disc_scaler = torch.cuda.amp.GradScaler()

        if config.load_model:
            self.util.load_checkpoint(self.config.gen_path, self.generator, self.opt_gen, self.config.gen_lr)
            self.util.load_checkpoint(self.config.disc_path, self.discriminator, self.opt_disc, self.config.disc_lr)
        else:
            self.util.init_directory()

        if self.config.use_wandb:
            wandb.init(project=f"outpainting-pix2pix", entity=self.config.wandb_entity, config=self.config)
            wandb.watch(self.generator)
            wandb.watch(self.discriminator)

    def train(self, train_loader, get_val_loader, epoch):
        looper = tqdm(train_loader)
        device = self.config.device

        for idx, (X, y) in enumerate(looper):
            try:
                X = X.to(device=device, non_blocking=True)
                y = y.to(device=device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    y_fake = self.generator(X)

                    d_real = self.discriminator(X, y)
                    d_fake = self.discriminator(X, y_fake.detach())

                    d_real_loss = self.bce(d_real, torch.ones_like(d_real))
                    d_fake_loss = self.bce(d_fake, torch.zeros_like(d_fake))
                    d_loss = (d_real_loss + d_fake_loss)
                
                self.opt_disc.zero_grad()
                self.disc_scaler.scale(d_loss).backward()
                self.disc_scaler.step(self.opt_disc)
                self.disc_scaler.update()

                with torch.cuda.amp.autocast():
                    d_fake = self.discriminator(X, y_fake)

                    g_fake_loss = self.bce(d_fake, torch.ones_like(d_fake))
                    l1_extra_loss = self.l1_loss(y_fake, y) * self.config.l1_lambda
                    g_loss = g_fake_loss + l1_extra_loss

                self.opt_gen.zero_grad()
                self.gen_scaler.scale(g_loss).backward()
                self.gen_scaler.step(self.opt_gen)
                self.gen_scaler.update()

                loss_dict = {
                    'g_loss': g_loss.item(),
                    'd_loss': d_loss.item()
                }

                if math.isnan(g_loss.item()) or math.isnan(d_loss.item()):
                    exit(-1, "Training produced nan losses, exitting...")
                    raise Exception("Training produced nan losses, exitting...")

                if (idx % self.config.log_batch_interval) == 0:
                    self.util.save_checkpoint(self.generator, self.opt_gen, self.config.gen_path)
                    self.util.save_checkpoint(self.discriminator, self.opt_disc, self.config.disc_path)

                    val_loader = get_val_loader(1)
                    self.util.save_examples(self.generator, val_loader, epoch, idx, self.config.example_path)
                
                if self.config.use_wandb and (idx % (self.config.log_batch_interval // 4)) == 0:
                    val_loader = get_val_loader(self.config.batch_size)
                    loss_dict['val_loss'] = self.util.calculate_validation_loss(self.generator, val_loader, self.l1_loss)
                    wandb.log(loss_dict)

                looper.set_postfix(loss_dict)
            except Exception as e:
                if self.config.break_on_error:
                    raise e
                else:
                    print(e)





