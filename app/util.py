import torch
from PIL import Image
import torchvision.transforms as transforms

def file2image(file):
    return Image.open(file)

def img2tensor(img):
    f = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return f(img)

def tensor2img(tensor):
    f = transforms.ToPILImage()
    return f(tensor[:3, :, :])