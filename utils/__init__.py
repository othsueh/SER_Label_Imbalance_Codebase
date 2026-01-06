import tomli
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
WANDB_TOKEN = os.getenv('WANDB_TOKEN')

if HUGGINGFACE_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HUGGINGFACE_TOKEN)
        print("Successfully logged in to Hugging Face Hub")
    except ImportError:
        print("Warning: huggingface_hub not installed. Skipping login.")
    except Exception as e:
        print(f"Warning: Failed to login to Hugging Face Hub: {e}")

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
