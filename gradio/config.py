import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '..')
load_dotenv()

from pth_config import SMALL, MEDIUM, LARGE

# Gradio
SERVER_NAME = '127.0.0.1'
SHARE = False
OUTPUT_DIR = './output/'

# Model
__MODEL = LARGE

INFERENCE_WEIGHTS_PATH = os.getenv('PTH_FILENAME', __MODEL.pth_filename)

# Configurations for model
PATCH_STREAM     = True                      # Stream training / inference
PATCH_SIZE       = 16                        # Patch Size
PATCH_LENGTH     = __MODEL.patch_len         # Patch Length
CHAR_NUM_LAYERS  = __MODEL.char_num_layers   # Number of layers in the decoder
PATCH_NUM_LAYERS = __MODEL.patch_num_layers  # Number of layers in the encoder
HIDDEN_SIZE      = __MODEL.hidden_size       # Hidden Size

# ====================================================================================================

TOP_K       = 9                                                   # Top k for sampling
TOP_P       = 0.9                                                 # Top p for sampling
TEMPERATURE = 1.2                                                 # Temperature for sampling
