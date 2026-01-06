from huggingface_hub import snapshot_download
from utils import config, HUGGINGFACE_TOKEN
import os

models = [
    # {"model": "wavlm-large", "source": "microsoft/wavlm-large"},
    {"model": "wavlm-base-plus", "source": "microsoft/wavlm-base-plus"}
    ]

if __name__ == "__main__":
    output_dir = config["PATH_TO_PRETRAINED_MODELS"]
    # output_dir = config["PATH_TO_SAVED_MODELS"]
    for m in models:
        new_dir = os.path.join(output_dir, m["model"])
        print(f'Checking {new_dir}')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        print('=' * 30 + f'\nDownloading {m["model"]}\n' + '=' * 30)
        snapshot_download(repo_id=m["source"],repo_type="model", local_dir=new_dir, token=HUGGINGFACE_TOKEN)
