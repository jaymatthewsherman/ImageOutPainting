import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
from .blocks import UNET

class Trainer:
    def __init__(self, config, util):
        self.config = config
        self.util = util

        self.unet = UNET(self.config).to(self.config.device)
        self.bce = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.unet.parameters(), lr=self.config.gen_lr)
        self.scaler = torch.cuda.amp.GradScaler()

        if self.config.load_model:
            self.util.load_checkpoint(self.config.gen_path, self.unet, self.optimizer, self.config.gen_lr)
        else:
            self.util.init_directory()
        
        if self.config.use_wandb:
            wandb.init(project=f"outpainting-unet", entity=self.config.wandb_entity, config=self.config, resume=self.config.load_model)
            wandb.watch(self.unet)

    def train(self, train_loader, get_val_loader, epoch):
        device = self.config.device
        looper = tqdm(train_loader)

        for idx, (X, y) in enumerate(looper):
            X = X.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            with torch.cuda.amp.autocast():
                y_hat = self.unet(X)
                loss = self.bce(y_hat, y)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_item = loss.item()
                if (idx % self.config.log_batch_interval) == 0:
                    wandb.log({"loss": loss_item})
                    
                    val_loader = get_val_loader(1)
                    self.util.save_checkpoint(self.unet, self.optimizer, self.config.gen_path)
                    self.util.save_examples(self.unet, val_loader, epoch, idx, self.config.example_path)
                looper.set_postfix(loss=loss_item)
