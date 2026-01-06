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
            from utils.hf_uploader import upload_model_to_hf
            # Note: utils.hf_uploader handles naming internally based on corpus/upstream/loss
            # Our parser takes full repo_id. 
            # To test the utility exactly as used in train.py, we should pass components that result in the repo_id.
            # But the user asked for "repo_id" validation...
            
            # If we strictly want to test the wrapper, we might need to deconstruct repo_id? 
            # Or we just use the wrapper and let it generate the name, but then we must know what it generates to download.
            
            # Actually, let's keep the manual upload logic HERE if we want to validate a specific repo_id passed by arg.
            # OR better: use the wrapper but override the params to match the args.repo_id if possible?
            # The wrapper does: repo_name = f"{corpus_name}_{upstream}_{loss_type}"
            # Repository = username/repo_name.
            
            # Let's trust the user wants to test the "mechanism".
            # If I use the wrapper, I force a naming convention.
            # Let's say we assume the user provides a repo_id that fits the convention OR we just test the upload function with dummy params and ignore the arg.repo_id for the *upload* part?
            
            # Wait, the user said "Wrap the upload method... and remember to setup login".
            # The validation script 'validate_hf_upload.py' was written by me. 
            # I should update it to use the new utility to PROVE the utility works.
            
            # Let's split the repo_id arg to feed into the function.
            full_repo = args.repo_id # e.g. user/MSP-PODCAST_wavlm-base-plus_InfoNCE
            if '/' in full_repo:
                username, repo_name = full_repo.split('/')
            else:
                repo_name = full_repo
            
            # Attempt to split repo_name by _ to guess components? 
            # That's validation specific logic. 
            # For simplicity, let's just CALL the function with kwargs that make sense.
            
            # Actually, to properly verify the utility, we should use it.
            # But the utility autogenerates the name.
            # Let's just use the utility logic directly? 
            # I'll modify the validation script to take separate args OR just hardcode dummy ones for the upload test?
            
            # Let's stick to using the utility, and PRINT what expected repo_id is.
            
            print("Using utils.hf_uploader.upload_model_to_hf...")
            # We need to match the signature: (corpus_name, upstream_model, loss_type, save_path)
            # We'll use dummy values that result in a predictable name, effectively ignoring args.repo_id for the GENERATION part if we want to test the tool exactly.
            # BUT args.repo_id is required for download.
            
            # Compromise: I will use the wrapper with specific values, and print what the resulting repo_id is, 
            # and ask the user to ensure args.repo_id matches it or update args.repo_id logic in script.
            
            # Actually, simpler: I'll use the API directly in validation script to keep it flexible for ANY repo_id,
            # BUT verify that I can import the utility.
            # The user's request was about "experiment_runner/train.py".
            # The validation script is a test artifact.
            
            # However, using the new utility here is a good integration test.
            
            # Let's define dummy vars:
            corpus = "TEST-CORPUS"
            upstream = "test_model"
            loss = "test_loss"
            
            upload_model_to_hf(corpus, upstream, loss, temp_dir)
            
            # The uploaded repo/URL will be username/TEST-CORPUS_test_model_test_loss
            # We override args.repo_id to this value for the download step.
            
            api = HfApi()
            username = api.whoami()['name']
            generated_repo_id = f"{username}/{corpus}_{upstream}_{loss}"
            print(f"Start using generated repo_id: {generated_repo_id}")
            args.repo_id = generated_repo_id 
            
        except ImportError:
             print("Could not import utils.hf_uploader or other dependency.")
             sys.exit(1)
        except Exception as e:
            print(f"Upload simulation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
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
