import torch, random
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from config import DEVICE, config


class Pix2PixUtil:
    def show(self, img):
        toPIL = transforms.ToPILImage()
        plt.imshow(toPIL(img[:3, :, :]))

    def save_checkpoint(self, model, optimizer, filename):
        # print("save_checkpoint()")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, checkpoint, model, optimizer, lr):
        # print("load_checkpoint()")

        """checkpoint = torch.load(checkpoint, map_location=DEVICE)"""
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def save_examples(self, model, loader, epoch, batch, directory):
        model.eval()
        for i, (X, y) in enumerate(loader):
            X = X.to(device=DEVICE)
            y_hat = model(X)

            if config.should_collapse:
                save_image(X, f"{directory}/epoch{epoch}_batch{batch}_{i}_input.png")
            save_image(y_hat, f"{directory}/epoch{epoch}_batch{batch}_{i}_pred.png")
            save_image(y, f"{directory}/epoch{epoch}_batch{batch}_{i}_real.png")
        model.train()


class UNetUtil:
    def show(self, img):
        toPIL = transforms.ToPILImage()
        plt.imshow(toPIL(img[:3, :, :]))

    def save_checkpoint(self, state, filename="checkpoint.pth.tar"):
        # print("save_checkpoint()")
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint, model):
        # print("load_checkpoint()")
        model.load_state_dict(checkpoint["state_dict"])

    def save_examples(self, loader, model, directory, device, epoch):
        model.eval()
        num_examples = 20
        for i, (X, y) in enumerate(loader):
            if random.random() > num_examples / len(loader):
                continue

            X = X.to(device=device)
            y_hat = model(X)
            X = X[:, :3, :, :]

            save_image(X, f"{directory}/epoch{epoch}_{i}_input.png")
            save_image(y_hat, f"{directory}/epoch{epoch}_{i}_pred.png")
            save_image(y, f"{directory}/epoch{epoch}_{i}_real.png")
        model.train()
