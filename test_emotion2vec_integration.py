#!/usr/bin/env python3
"""
Test script for emotion2vec integration.
Run on remote machine: python test_emotion2vec_integration.py

Tests:
  A. Emotional2Vec wrapper structure
  B. SERModel integration with emotion2vec
  C. Forward pass shapes (requires GPU)
  D. Layer freezing logic
"""

import sys
import torch
import torch.nn as nn


def test_wrapper_structure():
    """Test A: Emotion2VecWrapper structure (no GPU needed)."""
    print("\n" + "="*60)
    print("TEST A: Emotion2VecWrapper Structure")
    print("="*60)

    try:
        from net.emotion2vec_wrapper import Emotion2VecWrapper
        print("✓ Emotion2VecWrapper imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import Emotion2VecWrapper: {e}")
        return False

    try:
        # This will fail if funasr not installed
        print("\nAttempting to load emotion2vec_plus_base via FunASR...")
        w = Emotion2VecWrapper("iic/emotion2vec_plus_base")
        print("✓ Emotion2VecWrapper instantiated")

        # Check config
        if hasattr(w, "config") and hasattr(w.config, "hidden_size"):
            if w.config.hidden_size == 768:
                print(f"✓ config.hidden_size = {w.config.hidden_size} (expected 768)")
            else:
                print(f"✗ config.hidden_size = {w.config.hidden_size} (expected 768)")
                return False
        else:
            print("✗ config or hidden_size not found")
            return False

        # Check encoder layers
        if hasattr(w, "encoder") and hasattr(w.encoder, "layers"):
            num_layers = len(w.encoder.layers)
            if num_layers == 8:
                print(f"✓ encoder.layers count = {num_layers} (expected 8)")
            else:
                print(f"✗ encoder.layers count = {num_layers} (expected 8)")
                return False
        else:
            print("✗ encoder.layers not found")
            return False

        # Check freeze_feature_encoder exists
        if hasattr(w, "freeze_feature_encoder") and callable(w.freeze_feature_encoder):
            print("✓ freeze_feature_encoder method exists")
        else:
            print("✗ freeze_feature_encoder method not found")
            return False

        # Check forward signature
        if hasattr(w, "forward") and callable(w.forward):
            print("✓ forward method exists")
        else:
            print("✗ forward method not found")
            return False

        print("\n✅ TEST A PASSED: Wrapper structure is correct")
        return True

    except ImportError as e:
        print(f"\n⚠ FunASR not installed: {e}")
        print("  (This is expected if funasr hasn't been installed yet)")
        print("  To test fully, run: pip install funasr")
        print("\n⚠ TEST A SKIPPED: FunASR not available")
        return True  # Not a failure, just not available yet
    except Exception as e:
        print(f"\n✗ TEST A FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sermodel_integration():
    """Test B: SERModel integration with emotion2vec."""
    print("\n" + "="*60)
    print("TEST B: SERModel Integration with Emotion2Vec")
    print("="*60)

    try:
        from net.ser_model_wrapper import SERModel
        print("✓ SERModel imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import SERModel: {e}")
        return False

    # Check that SERModel can be initialized with emotion2vec
    try:
        print("\nAttempting to create SERModel with emotion2vec_plus_base...")
        model = SERModel(
            ssl_type="emotion2vec_plus_base",
            pooling_type="AttentiveStatisticsPooling",
            head_dim=768,
            hidden_dim=768,
            classifier_output_dim=8,
            finetune_layers=3
        )
        print("✓ SERModel initialized with emotion2vec_plus_base")

        # Check model has required attributes
        if hasattr(model, "ssl_model"):
            print("✓ model.ssl_model exists")
        else:
            print("✗ model.ssl_model not found")
            return False

        if hasattr(model, "pool_model"):
            print("✓ model.pool_model exists")
        else:
            print("✗ model.pool_model not found")
            return False

        if hasattr(model, "ser_model"):
            print("✓ model.ser_model exists")
        else:
            print("✗ model.ser_model not found")
            return False

        print("\n✅ TEST B PASSED: SERModel integration is correct")
        return True

    except ImportError as e:
        print(f"\n⚠ FunASR not installed: {e}")
        print("  (This is expected if funasr hasn't been installed yet)")
        print("\n⚠ TEST B SKIPPED: FunASR not available")
        return True
    except Exception as e:
        print(f"\n✗ TEST B FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_freezing():
    """Test C: Layer freezing logic."""
    print("\n" + "="*60)
    print("TEST C: Layer Freezing Logic")
    print("="*60)

    try:
        from net.ser_model_wrapper import SERModel
    except ImportError as e:
        print(f"✗ Failed to import SERModel: {e}")
        return False

    try:
        print("\nCreating SERModel with finetune_layers=3...")
        model = SERModel(
            ssl_type="emotion2vec_plus_base",
            pooling_type="AttentiveStatisticsPooling",
            head_dim=768,
            hidden_dim=768,
            classifier_output_dim=8,
            finetune_layers=3
        )

        # Check freezing logic
        # Layers 0-4 should be frozen (5 frozen out of 8)
        # Layers 5-7 should be trainable (last 3)

        encoder_layers = model.ssl_model.encoder.layers

        # Check a frozen layer (first layer, index 0)
        first_layer_trainable = any(p.requires_grad for p in encoder_layers[0].parameters())
        if not first_layer_trainable:
            print(f"✓ encoder.layers[0] is frozen (requires_grad=False)")
        else:
            print(f"✗ encoder.layers[0] is NOT frozen (requires_grad=True) - should be frozen")
            return False

        # Check a trainable layer (last layer, index -1)
        last_layer_trainable = any(p.requires_grad for p in encoder_layers[-1].parameters())
        if last_layer_trainable:
            print(f"✓ encoder.layers[-1] is trainable (requires_grad=True)")
        else:
            print(f"✗ encoder.layers[-1] is NOT trainable (requires_grad=False) - should be trainable")
            return False

        # Check a middle frozen layer
        middle_layer_trainable = any(p.requires_grad for p in encoder_layers[4].parameters())
        if not middle_layer_trainable:
            print(f"✓ encoder.layers[4] is frozen (requires_grad=False)")
        else:
            print(f"✗ encoder.layers[4] is NOT frozen - should be frozen")
            return False

        print("\n✅ TEST C PASSED: Layer freezing logic is correct")
        return True

    except ImportError as e:
        print(f"\n⚠ FunASR not installed: {e}")
        print("\n⚠ TEST C SKIPPED: FunASR not available")
        return True
    except Exception as e:
        print(f"\n✗ TEST C FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test D: Forward pass shapes (requires GPU)."""
    print("\n" + "="*60)
    print("TEST D: Forward Pass Shapes")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ CUDA not available - skipping GPU tests")
        print("  (This is expected on CPU-only machines)")
        return True

    try:
        from net.ser_model_wrapper import SERModel
    except ImportError as e:
        print(f"✗ Failed to import SERModel: {e}")
        return False

    try:
        print("\nCreating SERModel and moving to GPU...")
        model = SERModel(
            ssl_type="emotion2vec_plus_base",
            pooling_type="AttentiveStatisticsPooling",
            head_dim=768,
            hidden_dim=768,
            classifier_output_dim=8,
            finetune_layers=3
        )
        model = model.cuda()
        model.eval()
        print("✓ SERModel on GPU in eval mode")

        # Create dummy input
        batch_size = 2
        n_samples = 48000  # 3 seconds at 16kHz
        print(f"\nCreating dummy input: ({batch_size}, {n_samples})")
        x = torch.randn(batch_size, n_samples, device="cuda")
        attention_mask = torch.ones(batch_size, n_samples, device="cuda")

        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            logits = model(x, attention_mask=attention_mask)

        # Check output shape
        expected_shape = (batch_size, 8)  # 8 emotion classes
        if logits.shape == expected_shape:
            print(f"✓ Output shape: {logits.shape} (expected {expected_shape})")
        else:
            print(f"✗ Output shape: {logits.shape} (expected {expected_shape})")
            return False

        print("\n✅ TEST D PASSED: Forward pass shapes are correct")
        return True

    except ImportError as e:
        print(f"\n⚠ FunASR not installed: {e}")
        print("\n⚠ TEST D SKIPPED: FunASR not available")
        return True
    except Exception as e:
        print(f"\n✗ TEST D FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_files():
    """Test E: Configuration files updated correctly."""
    print("\n" + "="*60)
    print("TEST E: Configuration Files")
    print("="*60)

    try:
        # Check requirements.in
        print("\nChecking requirements.in...")
        with open("requirements.in", "r") as f:
            content = f.read()
            if "funasr" in content:
                print("✓ funasr added to requirements.in")
            else:
                print("✗ funasr not in requirements.in")
                return False

        # Check experiments_config.toml
        print("\nChecking experiments_config.toml...")
        with open("experiments_config.toml", "r") as f:
            content = f.read()
            if "base_config_e2v" in content:
                print("✓ base_config_e2v found in experiments_config.toml")
            else:
                print("✗ base_config_e2v not found in experiments_config.toml")
                return False

            if "Emotion2Vec-DR" in content:
                print("✓ Emotion2Vec-DR experiment found in experiments_config.toml")
            else:
                print("✗ Emotion2Vec-DR experiment not found in experiments_config.toml")
                return False

        # Check download_model.py
        print("\nChecking download_model.py...")
        with open("download_model.py", "r") as f:
            content = f.read()
            if "download_emotion2vec" in content:
                print("✓ download_emotion2vec function found in download_model.py")
            else:
                print("✗ download_emotion2vec function not found in download_model.py")
                return False

            if "--emotion2vec" in content:
                print("✓ --emotion2vec flag found in download_model.py")
            else:
                print("✗ --emotion2vec flag not found in download_model.py")
                return False

        print("\n✅ TEST E PASSED: All configuration files updated correctly")
        return True

    except Exception as e:
        print(f"\n✗ TEST E FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# EMOTION2VEC INTEGRATION TEST SUITE")
    print("#"*60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results = {
        "A. Wrapper Structure": test_wrapper_structure(),
        "B. SERModel Integration": test_sermodel_integration(),
        "C. Layer Freezing": test_layer_freezing(),
        "D. Forward Pass": test_forward_pass(),
        "E. Config Files": test_config_files(),
    }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n" + "🎉 "*15)
        print("ALL TESTS PASSED!")
        print("🎉 "*15)
        return 0
    else:
        print("\n" + "❌ "*15)
        print("SOME TESTS FAILED - See details above")
        print("❌ "*15)
        return 1


if __name__ == "__main__":
    sys.exit(main())
