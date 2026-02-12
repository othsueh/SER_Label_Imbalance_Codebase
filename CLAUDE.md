# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

200AI is a **Speech Emotion Recognition (SER)** system that fine-tunes a WavLM upstream model for emotion classification on the MSP-PODCAST corpus. The model performs categorical emotion prediction while also supporting dimensional emotion regression (arousal-valence).

## Development Setup

### Prerequisites
- CUDA-capable GPU (required - code asserts CUDA availability)
- Python 3.10+
- Poetry or pip for dependency management

### Installation & Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (required for Hugging Face & WandB integration)
cp .env.example .env
# Edit .env with your tokens:
# - HUGGINGFACE_TOKEN: for model access
# - WANDB_TOKEN: for experiment tracking
# - GIT_AUTH_TOKEN: optional, for git operations
```

### Key Commands

**Run experiments as defined in experiments_config.toml:**
```bash
python main.py
```

**Run in test/evaluation mode (uses run_evaluate instead of run_train):**
```bash
python main.py --test
```

**Download pretrained models:**
```bash
python download_model.py
```

**Process labels for categorical emotion:**
```bash
python process_labels_for_categorical.py
```

## Architecture Overview

### Data Flow
1. **Config Loading** → `utils/__init__.py` loads `config.toml` (dataset paths) and `experiments_config.toml` (experiment definitions)
2. **Dataset** → `dataset_module/ser_dataset.py` loads audio from disk and applies preprocessing
3. **Model** → `net/ser_model_wrapper.py` wraps the upstream WavLM model with custom heads
4. **Training/Eval** → `experiment_runner/train.py` and `experiment_runner/evaluate.py` orchestrate the training loop

### Directory Structure

**`net/`** — Neural network components
- `ser_model_wrapper.py` — Main model wrapper that combines upstream model + emotion prediction heads
- `loss_modules.py` — Custom losses: `BalancedSoftmaxLoss`, `CCCLoss`, etc.
- `pooling.py` — Audio feature pooling strategies (e.g., AttentiveStatisticsPooling)
- `ser.py` — Downstream emotion classification head

**`dataset_module/`** — Dataset handling
- `ser_dataset.py` — `SERDataset` class that loads MSP-PODCAST audio and labels

**`utils/`** — Utilities
- `__init__.py` — Config loading (TOML files), HF/WandB authentication
- `dataset/` — Data preprocessing, normalization, collation functions
- `data/` — Audio loading modules (wav.py, podcast.py)
- `hf_uploader.py` — Upload trained models to Hugging Face Hub
- `loss_manager.py` — Loss tracking and aggregation

**`experiment_runner/`** — Experiment orchestration
- `train.py` — `run_train()` handles full training and validation pipeline with early stopping and WandB logging
- `evaluate.py` — `run_evaluate()` runs evaluation on test sets
- `validate_hf_upload.py` — Validates model uploads to HF

**`configs/`** — Default configuration files
- `config_cat.json` — Categorical emotion config
- `config_dim.json` — Dimensional (arousal-valence) config

## Configuration System

### `config.toml` (Dataset & Path Configuration)
Maps corpus names to dataset paths:
```toml
[MSP-PODCAST]
PATH_TO_LABEL = '/path/to/labels.csv'
PATH_TO_AUDIO = '/path/to/audio'
PATH_TO_DATASET = '/path/to/webdataset'
```

### `experiments_config.toml` (Experiment Definitions)
Defines base config and experiment list. Example:
```toml
[base_config]
corpus = "MSP-PODCAST"
upstream_model = "wavlm-base-plus"
epoch = 30
batch_size = 16
learning_rate = 1e-5
loss_type = "WeightedCrossEntropy"

[[experiments]]
name = "UpstreamFinetune-Reweight"
model_type = "UpstreamFinetune"
config = "base_config"
config_update = { loss_type = "WeightedCrossEntropy" }
```

## Key Concepts

**Model Types:** The `model_type` parameter in experiments determines which model architecture to use (e.g., "UpstreamFinetune", "UpstreamGender").

**Loss Functions:** Supports multiple loss types via `get_loss_module()`:
- `BalancedSoftmaxLoss` — Handles class imbalance by re-weighting
- `WeightedCrossEntropy` — Standard weighted cross-entropy
- `CCCLoss` — For dimension regression (arousal-valence)

**Pooling:** Audio features from WavLM are pooled before classification:
- `AttentiveStatisticsPooling` — Learnable weighted pooling

**Mixed Precision:** AMP (Automatic Mixed Precision) enabled by default in `run_train()` for faster training and memory efficiency.

**Hybrid Early Stopping:** Training uses two stopping conditions:
1. **Loss Threshold Stop:** Stops immediately if current validation loss exceeds minimum validation loss × 1.05 (5% degradation)
2. **Smoothed Macro-F1 Stop:** Stops immediately if Simple Moving Average (SMA) of validation macro-F1 doesn't improve
- **Model Saving:** The model with the best smoothed (SMA) macro-F1 is saved, NOT the model with best loss. This prevents bias toward head classes.
- **Configuration:** Set `window` parameter (default: 5) in `experiments_config.toml` to control the SMA window size.

**Experiment Tracking:** All runs logged to W&B with config, metrics, and tags.

## Important Implementation Details

1. **CUDA Requirement:** Code asserts CUDA availability at startup (`torch.cuda.is_available()`). CPU-only runs will fail.

2. **WandB Integration:** `run_train()` calls `wandb.login()` and `wandb.init()` — ensure `WANDB_TOKEN` is set.

3. **Config Resolution:** Dataset paths are resolved from `config.toml` via the `corpus` parameter in experiments.

4. **Audio Loading:** Audio is loaded as `.wav` files and preprocessed (sample rate: 16kHz hardcoded in `experiment.py`).

5. **Metrics Tracked:** F1-score (macro/weighted), accuracy, unweighted accuracy rate (UAR), CCC (for dimensions), with head/mid/tail group stratification.

6. **Hybrid Early Stopping Details:**
   - SMA window fills gradually: uses average of available data until window is full
   - Loss threshold (1.05x) provides safety against catastrophic degradation
   - Smoothed F1 reduces noise from epoch-to-epoch variance
   - Stopping is strict: training stops immediately when smoothed F1 doesn't improve
   - Model saving occurs only when smoothed F1 improves
   - Backward compatible: old configs using `patience` will work (with deprecation warning)

## Common Debugging

- **"CUDA is not available" error:** Run on a GPU machine or modify the assertion in `main.py`.
- **"config.toml not found":** Working directory must be repository root.
- **Dataset path errors:** Verify `config.toml` paths match your environment (workspace, /datas paths, etc.).
- **WandB login fails:** Ensure `WANDB_TOKEN` is valid and set in `.env`.
- **Model download fails:** Check `HUGGINGFACE_TOKEN` and internet connectivity.

## Testing & Validation

Use `python main.py --test` to run evaluation mode, which calls `run_evaluate()` instead of training. This uses the same experimental configuration but skips training.
