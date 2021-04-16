import streamlit as st
import pandas as pd
import numpy as np
import torch

# ============= Project-level Imports ===========

from app import input_type_radio, input_select_images, input_upload_images, input_mask_direction
from app import file2image, tensor2img, img2tensor
from app import information
from model import default_config, Util, TopStripeMaskGenerator, RightStripeMaskGenerator, MaskApplier, Pix2PixEvaluator
from model import load_filepaths

tsmg = TopStripeMaskGenerator(default_config)
rsmg = RightStripeMaskGenerator(default_config)
mask_applier = MaskApplier(default_config)

default_config.saved_path = "./model/saved"

def load_model(model_name):
    default_config.model_name = model_name
    default_config.calculate_paths()

    util = Util(default_config)
    return Pix2PixEvaluator(default_config, util)

# Change model name to the one that's being tested
model_name = "parker"
model = load_model(model_name)

validation_path = "./app/examples/"
validation_files = [
    "mountainvillage.png", 
    "restaurant.png", 
    "kitchen.png", 
    "hotel.png",
    "wave.png",
    "rocking_chair.png",
    "desert.png",
    "bus_interior.png",
    "display.png"
]

# ========== Site Begin =============

st.title('Image Outpainting')
st.text('Parker Griep & Jay Sherman')

st.header('Demo')

# Image Selection
selecting = input_type_radio(st)
if selecting == "Select Image":
    img_file = validation_path + input_select_images(st, validation_files)
else:
    img_file = input_upload_images(st)

# Image Manipulation
if img_file is not None:

    # convert image file to tensor
    img = file2image(img_file)
    X = img2tensor(img)
    img = tensor2img(X)

    # get and apply mask
    direction = input_mask_direction(st)
    mask = tsmg.generate()
    if direction == "down":
        mask = torch.rot90(mask, 2)
    elif direction == "right":
        mask = rsmg.generate()
    elif direction == "left":
        mask = rsmg.generate()
        mask = torch.rot90(mask, 2)

    X = X[:3, :, :]
    X = mask_applier.apply(mask, X)
    masked_img = tensor2img(X)

    st.subheader("Ground Truth | Input Image")
    st.image([img, masked_img])

    y = model.predict(X.unsqueeze(0)).squeeze()
    predicted_img = tensor2img(y)

    st.subheader("Predicted Image")
    st.image(predicted_img)

    information.website_info(st)
    information.model_info(st)
    
