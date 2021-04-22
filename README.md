# Image Outpainting
A Pix2Pix-based neural network solution to image outpainting.

## To use the website
*Install streamlit if you haven't already: https://streamlit.io/
*Place the model parameters in the correct file location (without changing the code, this will be /'model//saved//parker//gen_checkpoint.pth.tar/' and /'model//saved//parker//disc_checkpoint.pth.tar/' for the generator and discriminator, respectfully. If you do not have model parameters, you will need to run the model once to get them, as the files are too large to upload to the remote repository.
*In the command line, navigate to where the project is located and run /'streamlit run site.py/'
*Select one of the example images from the dropdown menu, or upload an image from your computer. The model can handle images of any size, but expect high-resolution images to take more time.
*Select which side to outpaint, using the second dropdown menu.
*Observe the generated image below!

## To run the model
*Install dependencies (listed below).
*Set the values in /'model/config.py/' to reflect how you want to run the model. See the list of features below. Alternatively, many of these features can be set during the command call line below. Type /'--help/' for options.

### Hyperparameters in /'model/config.py/':
*MODEL_TYPE: the string representing the model, currently only processes "pix2pix"
*GEN_LEARNING_RATE: the learning rate for the generator in the Pix2Pix model
*DISC_LEARNING_RATE: the learning rate for the discriminator in the Pix2Pix model
*L1_LAMBDA: the lambda value for the Pix2Pix model
*DISC_LAMBDA: the lambda value for the discriminator in the Pix2Pix model
*DEVICE: device to run the model on ('cuda' if there is a cuda device available)
*BATCH_SIZE: how many images to include in a batch for training
*NUM_WORKERS: the number of cores to use for training
*PIN_MEMORY: 
*DATA_LIM:
*SHUFFLE: whether to shuffle the images during training or not
*ENTROPY_FILEPATHS: the relative file location of the list of images from least to greatest entropy
*TRAIN_FP: the realtive file location of the train images
*VAL_FP: the relative file location of the validation images
*FP_PREFIX: the relative file location of image data in general
*DEFAULT_EPOCHS: the number of epochs to train with
*DEFAULT_STRIPE_WIDTH: the width of the stripe to predict images for
*DEFAULT_PIC_DIM: a tuple of the dimensions of the image, being (number of channels, width, height)
*DEFAULT_COLOR: the color to mask the stripe with
*DEFAULT_COLLAPSE: whether to collapse the neural net layers during Pix2Pix
*DEFAULT_OUTSIDE: whether to test the model on outside images (if True, trains the model; if False, returns the model after potentially loading parameters)
*LOAD_MODEL: whether to load model parameters
*BREAK_ON_ERROR: whether to break if encountering nan while training
*LOG_BATCH_INTERVAL: how many batches between logging
*SAVED_PATH: the location of saved model parameters
*MODEL_NAME: the file name of the saved model's parameters
*USE_WANDB: whether to track hyperparameters with wandb
*WANDB_ENTITY: the username for wandb to associate with the hyperparameter sweeps

### DEPENDENCIES:
*torch
*torchvision
*secrets
*PIL
*argparse
*tqdm
*traceback
*wandb
*streamlit
