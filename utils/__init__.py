import tomli
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
WANDB_TOKEN = os.getenv('WANDB_TOKEN')

try:
    with open("config.toml", "rb") as f:
        config = tomli.load(f)
except FileNotFoundError:
    print("Warning: config.toml not found")
    config = {}

try:
    with open("experiments_config.toml", "rb") as f:
        experiments_config = tomli.load(f)
except FileNotFoundError:
    print("Warning: experiments_config.toml not found")
    experiments_config = {}

# from .data import *
# from .dataset import *
# from .loss_manager import *
# from .etc import *
