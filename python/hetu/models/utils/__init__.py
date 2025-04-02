CONFIG_NAME = "config.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
TORCH_WEIGHTS_NAME = "pytorch_model.bin"
TORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

from .config_utils import *
from .hub import *
from .model_utils import *
from .common_utils import *