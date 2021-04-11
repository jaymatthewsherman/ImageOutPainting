from config import *
import argparse

parser = argparse.ArgumentParser(description="Outpainting Model Trainer")
parser.add_argument("--gen_lr", "-glr", type=float, default=GEN_LEARNING_RATE, help="generator learning rate")
parser.add_argument("--disc_lr", "-dlr", type=float, default=DISC_LEARNING_RATE, help="discriminator learning rate")
parser.add_argument("--l1_lambda", "-l1", type=float, default=L1_LAMBDA, help="l1 loss lambda")
parser.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS, help="number of epochs")
parser.add_argument("--data_limit", "-lim", type=int, default=DATA_LIM, help="number of training images per epoch")
parser.add_argument("--load_model", "-load", default=False, action="store_true", help="whether or not to load an existing model")

parser.add_argument("--num_workers", "-nw", type=int, default=NUM_WORKERS, help="number of workers")
parser.add_argument("--batch_size", "-bs", type=int, default=BATCH_SIZE, help="batch size")
parser.add_argument("name", type=str, help="model name")


def config_from_args():
    args = parser.parse_args()

    return Config(
        epochs = args.epochs,
        gen_lr = args.gen_lr,
        disc_lr = args.disc_lr,
        l1_lambda = args.l1_lambda,
        data_lim = args.data_limit,
        model_name = args.name,
        load_model = args.load_model
    )

