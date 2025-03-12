import sys
sys.path.insert(0, '..')

from pth_config import SMALL, MEDIUM, LARGE

__MODEL = SMALL

# Configuration for the data
DATA_TRAIN_INDEX_PATH = ""
DATA_EVAL_INDEX_PATH  = ""

# Path of pretrained weights
PRETRAINED_PATH = "../pretrain/" + __MODEL.pth_filename

# Configuration for the model
PATCH_STREAM = True                                             # Stream training / inference
PATCH_SIZE = 16                                                 # Patch Size
PATCH_LENGTH = __MODEL.patch_len                                # Patch Length
CHAR_NUM_LAYERS = __MODEL.char_num_layers                       # Number of layers in the decoder
PATCH_NUM_LAYERS = __MODEL.patch_num_layers                     # Number of layers in the encoder
HIDDEN_SIZE = __MODEL.hidden_size                               # Hidden Size

# Configuration for the training
BATCH_SIZE = 1         
LEARNING_RATE = 1e-5   
NUM_EPOCHS = 64                                                 # Number of epochs to train for (if early stopping doesn't intervene)
ACCUMULATION_STEPS = 1                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full conaudio
LOAD_FROM_CHECKPOINT = False                                    # Whether to load weights from a checkpoint
WANDB_LOGGING = False                                           # Whether to log to wandb
WANDB_KEY = '<your_wandb_key>'

EXP_TAG = ''                                                    # Experiment tag for name differentiation
NAME =  EXP_TAG + \
        "_p_size_" + str(PATCH_SIZE) + \
        "_p_length_" + str(PATCH_LENGTH) + \
        "_p_layers_" + str(PATCH_NUM_LAYERS) + \
        "_c_layers_" + str(CHAR_NUM_LAYERS) + \
        "_h_size_" + str(HIDDEN_SIZE) + \
        "_lr_" + str(LEARNING_RATE) + \
        "_batch_" + str(BATCH_SIZE)

WEIGHTS_PATH = "weights_notagen_" + NAME + ".pth"                  # Path to save weights
LOGS_PATH    = "logs_notagen_"    + NAME + ".txt"                     # Path to save logs
WANDB_NAME = NAME
