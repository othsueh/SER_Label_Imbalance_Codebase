from huggingface_hub import snapshot_download
from utils import config, HUGGINGFACE_TOKEN
import argparse
import os

models = [
    # {"model": "wavlm-large", "source": "microsoft/wavlm-large"},
    {"model": "wavlm-base-plus", "source": "microsoft/wavlm-base-plus"}
    ]


def download_emotion2vec():
    """Pre-download emotion2vec_plus_base via FunASR to ~/.cache/modelscope."""
    try:
        from funasr import AutoModel as FunASRAutoModel
        print("Pre-downloading emotion2vec_plus_base via FunASR...")
        FunASRAutoModel(model="iic/emotion2vec_plus_base", device="cpu")
        print("Done. Cached to ~/.cache/modelscope.")
    except ImportError:
        print("funasr not installed. Run: pip install funasr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained models")
    parser.add_argument("--emotion2vec", action="store_true", help="Download emotion2vec_plus_base via FunASR")
    args = parser.parse_args()

    if args.emotion2vec:
        download_emotion2vec()
    else:
        output_dir = config["PATH_TO_PRETRAINED_MODELS"]
        # output_dir = config["PATH_TO_SAVED_MODELS"]
        for m in models:
            new_dir = os.path.join(output_dir, m["model"])
            print(f'Checking {new_dir}')
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            print('=' * 30 + f'\nDownloading {m["model"]}\n' + '=' * 30)
            snapshot_download(repo_id=m["source"],repo_type="model", local_dir=new_dir, token=HUGGINGFACE_TOKEN)
