import os
from huggingface_hub import HfApi, create_repo

def upload_model_to_hf(corpus_name, upstream_model, loss_type, save_path):
    """
    Uploads the trained model to Hugging Face Hub.
    
    Args:
        corpus_name (str): Name of the corpus (e.g., 'MSP-PODCAST')
        upstream_model (str): Name of the upstream model (e.g., 'wavlm-base-plus')
        loss_type (str): Type of loss used (e.g., 'WeightedCrossEntropy')
        save_path (str): Path to the saved model directory
    """
    try:
        print("\n" + "="*40)
        print("Initiating Hugging Face Hub Upload...")
        
        # Construct model name: corpus + model + loss_type
        # Clean naming: replace / with _ in model name
        upstream = upstream_model.replace('/', '_')
        
        repo_name = f"{corpus_name}_{upstream}_{loss_type}"
        
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']
        repo_id = f"{username}/{repo_name}"
        
        print(f"Target Repo ID: {repo_id}")
        
        # Create repo (private=True by default to be safe)
        create_repo(repo_id, private=True, exist_ok=True)
        
        # Upload
        if os.path.exists(save_path) and os.path.isdir(save_path):
            print(f"Uploading files from {save_path} ...")
            api.upload_folder(
                folder_path=save_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"Upload complete! Model available at https://huggingface.co/{repo_id}")
        else:
            print(f"Error: Save path {save_path} is not a directory or does not exist.")
            
    except ImportError:
        print("huggingface_hub library not found. Skipping upload.")
        print("Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"An error occurred during HF upload: {e}")
        # print("Ensure you are logged in with 'huggingface-cli login' or have HF_TOKEN set.") 
        # (This should be handled by utils/__init__.py now, but keeping error explicit is good)
    print("="*40 + "\n")
