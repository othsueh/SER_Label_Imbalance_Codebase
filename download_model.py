from huggingface_hub import snapshot_download
from utils import config, HUGGINGFACE_TOKEN
import argparse
import os

models = [
    # {"model": "wavlm-large", "source": "microsoft/wavlm-large"},
    {"model": "wavlm-base-plus", "source": "microsoft/wavlm-base-plus"}
    ]


def download_emotion2vec():
    """Pre-download emotion2vec_plus_base via FunASR to PATH_TO_PRETRAINED_MODELS."""
    try:
        from funasr import AutoModel as FunASRAutoModel
        import os

        output_dir = config["PATH_TO_PRETRAINED_MODELS"]
        emotion2vec_dir = os.path.join(output_dir, "emotion2vec_plus_base")
        os.makedirs(emotion2vec_dir, exist_ok=True)

        print(f"Pre-downloading emotion2vec_plus_base via FunASR to {emotion2vec_dir}...")
        # Set environment variable to control cache directory
        os.environ["MODELSCOPE_CACHE"] = emotion2vec_dir

        FunASRAutoModel(model="iic/emotion2vec_plus_base", device="cpu")
        print(f"Done. Cached to {emotion2vec_dir}.")
    except ImportError:
        print("funasr not installed. Run: pip install funasr")
    except Exception as e:
        print(f"Error downloading emotion2vec: {e}")


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
