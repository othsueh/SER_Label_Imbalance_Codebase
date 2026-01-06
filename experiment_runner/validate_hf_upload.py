import argparse
import sys
import os
from huggingface_hub import snapshot_download

# Add root to sys.path to ensure imports work if run from root
sys.path.append(os.getcwd())

try:
    from experiment_runner.evaluate import run_evaluate
    import utils
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please run this script from the project root directory, e.g.: python experiment_runner/validate_hf_upload.py ...")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Validate HF Model Upload by downloading and running evaluation.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID (e.g., username/test_model_123)")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split to evaluate on (test/dev)") # Default to dev for quick check
    parser.add_argument("--skip_upload", action="store_true", help="Skip the creation and upload step, only validate download")
    
    args = parser.parse_args()
    
    # --- Step 1: Initialize, Save, and Upload (Mocking train.py) ---
    if not args.skip_upload:
        print("="*40)
        print("Step 1: Simulating Training Finish & Upload...")
        try:
            from net.ser_model_wrapper import SERModel
            from huggingface_hub import HfApi, create_repo
            import tempfile
            import shutil

            # 1. Initialize Dummy Model
            print("Initializing dummy SERModel...")
            # Using small params for speed
            model = SERModel(
                ssl_type="microsoft/wavlm-base-plus", 
                pooling_type="AttentiveStatisticsPooling",
                head_dim=128,
                hidden_dim=128,
                classifier_output_dim=4, # Dummy class count
                dropout=0.1,
                finetune_layers=0
            ) 
            
            # 2. Save Locally
            temp_dir = tempfile.mkdtemp()
            print(f"Saving model to temporary path: {temp_dir}")
            model.save_pretrained(temp_dir)
            
            # 3. Upload
            print(f"Uploading to HF Hub: {args.repo_id}...")
            api = HfApi()
            create_repo(args.repo_id, private=True, exist_ok=True)
            
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=args.repo_id,
                repo_type="model"
            )
            print("Upload successful!")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"Upload simulation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # --- Step 2: Download and Evaluate ---
    print("\n" + "="*40)
    print(f"Step 2: Validation - Downloading model from {args.repo_id}...")
    try:
        model_path = snapshot_download(repo_id=args.repo_id)
        print(f"Model downloaded to: {model_path}")
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("Ensure you have access to the repo and standard HF credentials.")
        sys.exit(1)
        
    print("\nStarting Evaluation...")
    
    # Resolve Corpus from local config to ensure we load the right dataset
    # We rely on the local environment having the dataset paths configured in config.toml/utils
    corpus = "MSP-PODCAST" # Default
    if hasattr(utils, 'experiments_config') and 'base_config' in utils.experiments_config:
        corpus = utils.experiments_config['base_config'].get('corpus', corpus)
    
    print(f"Using Corpus: {corpus}")
    print("Delegate to experiment_runner.evaluate.run_evaluate...")
    
    # run_evaluate will detect config.json in model_path and use SERModel.from_pretrained
    try:
        run_evaluate(
            model_type="Validation",
            model_path=model_path,
            corpus=corpus,
            split=args.split,
            batch_size=32
        )
        print("\nValidation Run Complete: Success!")
    except Exception as e:
        print(f"\nValidation process failed: {e}")
        import traceback
        traceback.print_exc()
    print("="*40)

if __name__ == "__main__":
    main()
