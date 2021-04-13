import torch, secrets
from PIL import ImageColor

# Hyperparameters
MODEL_TYPE = "pix2pix"

GEN_LEARNING_RATE = 2e-4
DISC_LEARNING_RATE = 2e-4
L1_LAMBDA = 100
DISC_LAMBDA = 200

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 12
NUM_WORKERS = 2
PIN_MEMORY = True
DATA_LIM = 36_000
SHUFFLE = False

ENTROPY_FILEPATHS = "./dataset/entropy-filepaths-ordered.txt"
TRAIN_FP = "./dataset/places365_standard/train/"
VAL_FP = "./dataset/places365_standard/val/"
FP_PREFIX = "./dataset/"

DEFAULT_EPOCHS = 100
DEFAULT_STRIPE_WIDTH = 12
DEFAULT_PIC_DIM = (3, 256, 256)
DEFAULT_COLOR = "#000000"
DEFAULT_COLLAPSE = False
DEFAULT_OUTSIDE = True

LOAD_MODEL = False
BREAK_ON_ERROR = True
LOG_BATCH_INTERVAL = 100

SAVED_PATH = "./saved"

MODEL_NAME = None

USE_WANDB = True
WANDB_ENTITY = "buntry"

class Config:
    def __init__(self,
        model_type=MODEL_TYPE,
        epochs=DEFAULT_EPOCHS,
        entropy_fp=ENTROPY_FILEPATHS,
        fp_prefix=FP_PREFIX,
        stripe_width=DEFAULT_STRIPE_WIDTH, 
        pic_dim=DEFAULT_PIC_DIM,
        color=DEFAULT_COLOR,
        should_collapse=DEFAULT_COLLAPSE,
        outside=DEFAULT_OUTSIDE,
        gen_lr = GEN_LEARNING_RATE,
        disc_lr = DISC_LEARNING_RATE,
        l1_lambda = L1_LAMBDA,
        disc_lambda = DISC_LAMBDA,
        data_lim = DATA_LIM,
        model_name = MODEL_NAME,
        saved_path = SAVED_PATH,
        use_wandb = USE_WANDB,
        shuffle = SHUFFLE,
        device = DEVICE,
        pin_memory = PIN_MEMORY,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        train_fp = TRAIN_FP,
        val_fp = VAL_FP,
        wandb_entity = WANDB_ENTITY,
        load_model = LOAD_MODEL,
        log_batch_interval = LOG_BATCH_INTERVAL,
        break_on_error = BREAK_ON_ERROR,
        overwrite = False
    ):
        self.type = model_type
        self.epochs = epochs
        self.entropy_fp = entropy_fp
        self.fp_prefix = fp_prefix
        self.stripe_width = stripe_width
        self.pic_dim = list(pic_dim)
        self.pic_channels = self.pic_dim[0]
        self.pic_height = self.pic_dim[1]
        self.pic_width = self.pic_dim[2]
        self.color = [c / 256 for c in ImageColor.getcolor(color, "RGB")]
        self.should_collapse = should_collapse
        self.outside = outside
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.l1_lambda = l1_lambda
        self.disc_lambda = disc_lambda
        self.data_lim = data_lim
        self.model_name = model_name
        self.use_wandb = use_wandb
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fp = train_fp
        self.val_fp = val_fp
        self.device = device
        self.wandb_entity = wandb_entity
        self.load_model = load_model
        self.overwrite = overwrite
        self.log_batch_interval = log_batch_interval
        self.break_on_error = break_on_error

        self.saved_path = saved_path
        self.calculate_paths()

    def calculate_paths(self):
        self.gen_path = f"{self.saved_path}/{self.model_name}/gen_checkpoint.pth.tar"
        self.disc_path = f"{self.saved_path}/{self.model_name}/disc_checkpoint.pth.tar"
        self.example_path = f"{self.saved_path}/{self.model_name}/examples/"

default_config = Config()