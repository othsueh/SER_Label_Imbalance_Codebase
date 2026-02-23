# Emotion2Vec Integration - Implementation Summary

## Overview

This document summarizes the integration of emotion2vec as an upstream model for the Speech Emotion Recognition (SER) system. The emotion2vec model is loaded via FunASR and wrapped to provide a HuggingFace-compatible interface.

## What Was Implemented

### 1. **New File: `net/emotion2vec_wrapper.py`**
   - Created `Emotion2VecWrapper` class that adapts FunASR emotion2vec to the HuggingFace interface
   - Provides compatibility layer for:
     - `.config.hidden_size` → 768
     - `.encoder.layers` → nn.ModuleList of 8 transformer blocks
     - `.freeze_feature_encoder()` → freezes CNN local_encoder
     - `forward(input_values, attention_mask)` → returns last_hidden_state

### 2. **Modified: `net/ser_model_wrapper.py`**
   - Added emotion2vec detection guard (line 14-20)
   - Conditionally loads either HuggingFace AutoModel or Emotion2VecWrapper
   - Handles both "emotion2vec_plus_base" and "iic/emotion2vec_plus_base" formats
   - No changes needed to layer freezing or forward pass logic (already compatible)

### 3. **Modified: `experiments_config.toml`**
   - Added `[base_config_e2v]` section with emotion2vec-specific settings
   - Added `[[experiments]]` entry for "Emotion2Vec-DR" experiment
   - Configuration mirrors WavLM baseline with appropriate emotion2vec parameters

### 4. **Modified: `requirements.in`**
   - Added `funasr` dependency (needed for emotion2vec loading)

### 5. **Modified: `download_model.py`**
   - Added `--emotion2vec` CLI flag for pre-downloading emotion2vec models
   - FunASR automatically caches to `~/.cache/modelscope`

## Key Technical Details

### Emotion2Vec Structure (FunASR)
- **Hidden size**: 768 (same as WavLM-base)
- **Encoder layers**: 8 transformer blocks (vs WavLM's 12 layers)
- **Frame rate**: 50Hz (320 samples/frame)
- **Freezing**: CNN local_encoder is frozen; transformer blocks can be fine-tuned

### API Mapping

| Operation | WavLM (HF) | Emotion2Vec (FunASR) |
|-----------|-----------|----------------------|
| Load | `AutoModel.from_pretrained()` | `funasr.AutoModel()` → `.model` |
| Config | `.config.hidden_size` | Wrapper: `_FakeConfig(768)` |
| Layers | `.encoder.layers` | Wrapper: `_FakeEncoder(._e2v.blocks)` |
| Freeze CNN | `.freeze_feature_encoder()` | Wrapper: freezes `modality_encoders["AUDIO"].local_encoder` |
| Forward | `(x, attention_mask).last_hidden_state` | Wrapper: `.extract_features()` → `{"x": ...}` |

### Fine-tuning Configuration
- **Finetune layers**: 3 (last 3 of 8 blocks = 37.5% vs WavLM's 25%)
- **Pooling**: AttentiveStatisticsPooling (unchanged)
- **Head**: EmotionRegression (unchanged)
- **Loss**: WeightedCrossEntropy with DR variant

## Setup & Usage

### Installation
```bash
# Install dependencies (including funasr)
pip install -r requirements.txt
# Or manually:
pip install funasr
```

### Pre-download (Optional)
```bash
python download_model.py --emotion2vec
# Downloads to ~/.cache/modelscope/iic/emotion2vec_plus_base
```

### Run Training
```bash
# Uses experiments_config.toml with Emotion2Vec-DR active
python main.py
```

### Run Evaluation Only
```bash
python main.py --test
```

## Testing

A comprehensive test suite is provided: `test_emotion2vec_integration.py`

### Run All Tests
```bash
python test_emotion2vec_integration.py
```

### Tests Included
1. **Test A**: Emotion2VecWrapper structure (config, layers, methods)
2. **Test B**: SERModel integration (model creation, attribute checks)
3. **Test C**: Layer freezing logic (first layer frozen, last 3 trainable)
4. **Test D**: Forward pass shapes (GPU required, optional)
5. **Test E**: Configuration files (requirements, experiments, download script)

### Expected Output
```
✅ TEST A PASSED: Wrapper structure is correct
✅ TEST B PASSED: SERModel integration is correct
✅ TEST C PASSED: Layer freezing logic is correct
✅ TEST D PASSED: Forward pass shapes are correct (or SKIPPED if no GPU)
✅ TEST E PASSED: All configuration files updated correctly

🎉 ALL TESTS PASSED!
```

## Verification Steps

### Manual Structural Check (no GPU)
```python
from net.emotion2vec_wrapper import Emotion2VecWrapper
w = Emotion2VecWrapper("iic/emotion2vec_plus_base")
print(w.config.hidden_size)    # Expected: 768
print(len(w.encoder.layers))   # Expected: 8
```

### SERModel Integration Check
```python
from net.ser_model_wrapper import SERModel
model = SERModel(ssl_type="emotion2vec_plus_base",
                 pooling_type="AttentiveStatisticsPooling",
                 head_dim=768, hidden_dim=768,
                 classifier_output_dim=8, finetune_layers=3)
# Verify freezing
print(any(p.requires_grad for p in model.ssl_model.encoder.layers[0].parameters()))  # False
print(any(p.requires_grad for p in model.ssl_model.encoder.layers[-1].parameters())) # True
```

### Forward Pass Check (requires GPU)
```python
model.cuda(); model.eval()
x = torch.randn(2, 48000).cuda()
mask = torch.ones(2, 48000).cuda()
with torch.no_grad():
    logits = model(x, attention_mask=mask)
print(logits.shape)  # Expected: (2, 8)
```

## Comparison: Emotion2Vec vs WavLM

| Aspect | WavLM-base | Emotion2Vec-base |
|--------|-----------|------------------|
| Hidden size | 768 | 768 |
| Layers | 12 | 8 |
| Fine-tune % | 25% (3/12) | 37.5% (3/8) |
| Training params | Fewer | Fewer |
| Pretraining | General speech SSL | Emotion-specific |
| Frame rate | 50Hz | 50Hz |

**Hypothesis**: emotion2vec_plus_base may achieve better SER results due to emotion-specific pretraining, though fewer total layers may require careful hyperparameter tuning.

## Troubleshooting

### Issue: "funasr is required for emotion2vec"
**Solution**: Install with `pip install funasr`

### Issue: FunASR downloads to `~/.cache/modelscope` instead of local directory
**Expected behavior**: FunASR caches models globally. The training script reads from cache automatically.

### Issue: "Cannot find transformer encoder layers"
**Solution**: Check FunASR version. Current implementation assumes `blocks` attribute in Emotion2vec. Fallback logic tries `encoder`, `transformer`, `layers` in sequence with helpful error messages.

### Issue: Shape mismatch during training
**Solution**: Ensure audio is preprocessed to 16kHz. The wrapper assumes 16kHz input (same as WavLM).

## Files Changed

```
net/
  ├── emotion2vec_wrapper.py (NEW)
  └── ser_model_wrapper.py (MODIFIED)
experiments_config.toml (MODIFIED)
requirements.in (MODIFIED)
download_model.py (MODIFIED)
test_emotion2vec_integration.py (NEW)
EMOTION2VEC_INTEGRATION_SUMMARY.md (NEW)
```

## Next Steps

1. Run `python test_emotion2vec_integration.py` on your remote machine
2. Pre-download models: `python download_model.py --emotion2vec`
3. Run training: `python main.py` (uses Emotion2Vec-DR experiment)
4. Monitor W&B metrics and compare with WavLM baseline
5. Adjust `finetune_layers` if needed (try 2 for fair parameter comparison)

## Notes

- Emotion2vec loads from HuggingFace Hub (`emotion2vec/emotion2vec_plus_base`) via FunASR
- Models are cached in `~/.cache/modelscope/iic/emotion2vec_plus_base`
- The wrapper is fully compatible with existing SER training pipeline
- All losses, pooling, and heads work unchanged
- Early stopping and metrics tracking work unchanged
