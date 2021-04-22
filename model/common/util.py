import torch, os
from torchvision.utils import save_image

class Util:

    def __init__(self, config):
        self.config = config

    def save_checkpoint(self, model, optimizer, filename):
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, checkpoint, model, optimizer, lr):
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def save_examples(self, model, loader, epoch, batch, directory):
        model.eval()
        for i, (X, y) in enumerate(loader):
            X = X.to(device=self.config.device)
            y_hat = model(X)

            if self.config.should_collapse:
                save_image(X, f"{directory}/epoch{epoch}_batch{batch}_{i}_input.png")
            save_image(y_hat, f"{directory}/epoch{epoch}_batch{batch}_{i}_pred.png")
            save_image(y, f"{directory}/epoch{epoch}_batch{batch}_{i}_real.png")
        model.train()

    def init_directory(self):
        if not os.path.isdir(self.config.saved_path):
            os.mkdir(self.config.saved_path)
        if not os.path.isdir(f"{self.config.saved_path}/{self.config.model_name}"):
            os.mkdir(f"{self.config.saved_path}/{self.config.model_name}")
        elif not self.config.load_model and not self.config.overwrite:
            ans = input(f"Detected pre-existing model [{self.config.model_name}], would you like to overwrite it? [Y/n]")
            if ans in ["n", "N"]:
                exit(0)
        if not os.path.isdir(f"{self.config.saved_path}/{self.config.model_name}/examples"):
            os.mkdir(f"{self.config.saved_path}/{self.config.model_name}/examples")

    def calculate_validation_loss(self, model, loader, loss_fn):
        model.eval()
        loss = 0
        for _, (X, y) in enumerate(loader):
            X = X.to(device=self.config.device)
            y = y.to(device=self.config.device)

            y_hat = model(X)
            
            batch_loss = loss_fn(y, y_hat).item()
            
            if loss == 0:
                loss = batch_loss
            else:
                loss += batch_loss
        model.train()
        return loss