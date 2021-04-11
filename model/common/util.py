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

    def save_examples(self, model, get_loader, epoch, batch, directory):
        model.eval()
        loader = get_loader()
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
        if not os.path.isdir(f"{self.config.saved_path}/{self.config.model_name}/examples"):
            os.mkdir(f"{self.config.saved_path}/{self.config.model_name}/examples")