import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
import traceback
import math
import argparse

# library imports
from Pix2Pix.generator import Generator
from Pix2Pix.discriminator import Discriminator
from config import *
from UNet.blocks import UNET
from UNet.image_dataset import UNetImageDataset
from masking.mask_transforms import RandomBorderMaskTransform, ApplyMaskTransform, MaskedAreaTransform
from util import Pix2PixUtil, UNetUtil
from Pix2Pix.dataloader import get_train_loader, get_val_loader

u_p2p = Pix2PixUtil()
u_un = UNetUtil()

def train_unet(loader, model, optimizer, loss_function, scaler):
    looper = tqdm(loader)

    for idx, (X, y) in enumerate(looper):
        X = X.to(device=DEVICE, non_blocking=True)
        y = y.to(device=DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast():
            y_hat = model(X)
            loss = loss_function(y_hat, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # logging
        l = loss.item()
        if idx % LOG_BATCH_INTERVAL == 0:
            wandb.log({"loss": l})
        looper.set_postfix(loss=l)

def main_unet(config):
    wandb.init(project='outpainting-unet', entity='jaymatthewsherman')

    entropy_ordered_filepaths = UNetImageDataset.load_filepaths(ENTROPY_FILEPATHS, prefix=FP_PREFIX)
    entropy_ordered_filepaths = ["/".join(eof.split("/")[1:]) for eof in entropy_ordered_filepaths]
    img_dataset = UNetImageDataset(filepaths=entropy_ordered_filepaths, shuffle=True, lim=100_000,
                                   both=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        RandomBorderMaskTransform(config)]),
                                   xonly=ApplyMaskTransform(config),
                                   yonly=MaskedAreaTransform(config))
    train_loader = torch.utils.data.DataLoader(img_dataset,
                                               batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    unet = UNET(config).to(DEVICE).cuda()
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(unet.parameters(), lr=UNET_LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if LOAD_MODEL:
        u_un.load_checkpoint(torch.load("UNet/saved/checkpoint.pth.tar"), unet)

    wandb.watch(unet)
    for epoch in range(config.epochs):
        try:
            train_unet(train_loader, unet, optimizer, loss_function, scaler)

            checkpoint = {
                "state_dict": unet.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            u_un.save_checkpoint(checkpoint, "UNet/saved/checkpoint.pth.tar")
            u_un.save_examples(train_loader, unet, directory="UNet/saved", device=DEVICE, epoch=epoch)
        except RuntimeError as re:
            print("experienced runtime error")
            print(re)
            pass

def train_pix2pix(disc, gen, loader, opt_disc, opt_gen, disc_scaler, gen_scaler, l1_loss, bce, epoch):
    looper = tqdm(loader)

    for idx, (X, y) in enumerate(looper):
        try:
            #ensure that X and y do not have nan
            if (X != X).any() or (y != y).any():
                raise Exception(f"Started with nan values at {idx}")
            X = X.to(device=DEVICE, non_blocking=True)
            y = y.to(device=DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                y_fake = gen(X)

                d_real = disc(X, y)
                d_fake = disc(X, y_fake.detach())

                d_real_loss = bce(d_real, torch.ones_like(d_real))
                d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
                d_loss = (d_real_loss + d_fake_loss) / 2

            opt_disc.zero_grad()
            disc_scaler.scale(d_loss).backward()
            disc_scaler.step(opt_disc)
            disc_scaler.update()

            with torch.cuda.amp.autocast():
                d_fake = disc(X, y_fake)

                g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
                l1_extra_loss = l1_loss(y_fake, y) * L1_LAMBDA
                g_loss = g_fake_loss + l1_extra_loss
                
            wandb.log({"g_loss" : g_loss})

            opt_gen.zero_grad()
            gen_scaler.scale(g_loss).backward()
            gen_scaler.step(opt_gen)
            gen_scaler.update()

            loss_dict = {
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item()
            }

            if math.isnan(g_loss.item()) or math.isnan(d_loss.item()):
                raise Exception("Got nan values")

            if (idx % LOG_BATCH_INTERVAL) == 0:
                u_p2p.save_checkpoint(gen, opt_gen, GEN_CHECKPOINT_PATH)
                u_p2p.save_checkpoint(disc, opt_disc, DISC_CHECKPOINT_PATH)
                u_p2p.save_examples(gen, get_val_loader(config), epoch, idx, '../saved')
            if (idx % (LOG_BATCH_INTERVAL // 4)) == 0:
                wandb.log(loss_dict)

            looper.set_postfix(loss_dict)
        except Exception as e:
            with open(f"error_index_{idx}.txt", "w") as f:
                f.write("".join(traceback.format_exception(etype = type(e), value=e, tb=e.__traceback__)))
            if BREAK_ON_ERROR:
                raise e

def main_pix2pix(config):
    if not config.outside:
        wandb.init(project='outpainting-pix2pix', entity='jaymatthewsherman')

    gen = Generator(config, in_channels=4).to(DEVICE)
    disc = Discriminator(config, in_channels=7).to(DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.gen_lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.disc_lr, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    gen_scaler = torch.cuda.amp.GradScaler()
    disc_scaler = torch.cuda.amp.GradScaler()

    train_loader = get_train_loader(config)

    if LOAD_MODEL:
        u_p2p.load_checkpoint(GEN_CHECKPOINT_PATH, gen, opt_gen, GEN_LEARNING_RATE)
        u_p2p.load_checkpoint(DISC_CHECKPOINT_PATH, disc, opt_disc, DISC_LEARNING_RATE)

    if not config.outside:
        wandb.watch(gen)
        wandb.watch(disc)
        for epoch in range(config.epochs):
            train_pix2pix(disc, gen, train_loader, opt_disc, opt_gen, disc_scaler, gen_scaler, L1_LOSS, BCE, epoch)
            raise ValueError('not here')
    else:
        return gen

def main(config=config):

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Type of model to train- options are 'pix2pix' and 'unet'")
    parser.add_argument("--num_epochs", type=int, help="number of epochs to run")
    parser.add_argument("--unet_lr", type=float, help="learning rate for UNet")
    parser.add_argument("--disc_lr", type=float, help="learning rate for Pix2Pix generator")
    parser.add_argument("--gen_lr", type=float, help="learning rate for UNet")

    args = parser.parse_args()
    if args.num_epochs:
        config.epochs = args.num_epochs
    if args.unet_lr:
        config.unet_lr = args.unet_lr
    if args.gen_lr:
        config.gen_lr = args.gen_lr
    if args.disc_lr:
        config.disc_lr = args.disc_lr

    if not args.model:
        raise ValueError("Must choose `model` parameter! Options are 'pix2pix' and 'unet'.")

    if args.model == 'pix2pix':
        return main_pix2pix(config)
    elif args.model == 'unet':
        #set should_collapse to True- not meaningful for it to be False for UNet
        config.should_collapse = True
        return main_unet(config)
    else:
        raise ValueError(f"'{args.model}' is not a handled model type. Options are 'pix2pix' and 'unet'.")

if __name__ == "__main__":
    main()