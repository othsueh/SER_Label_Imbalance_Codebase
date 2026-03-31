# Investigating Class imbalance in Speech Emotion Recognition

Fine-tunes a [WavLM](https://arxiv.org/abs/2110.13900) upstream model for emotion classification on the [MSP-PODCAST](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) corpus.

## Setup

### Prerequisites

- CUDA-capable GPU
- Python 3.10+

### Install

```bash
pip install -r requirements.txt

cp .env.example .env
# Fill in:
# HUGGINGFACE_TOKEN — for model access
# WANDB_TOKEN       — for experiment tracking
```

### Download pretrained models

```bash
python download_model.py
```

## Usage

```bash
# Train experiments defined in experiments_config.toml
python main.py

# Evaluate only (no training)
python main.py --test
```

## Configuration

**`config.toml`** — Dataset paths (audio, labels, webdataset).

**`experiments_config.toml`** — Experiment definitions. Each experiment specifies a model type, loss function, and any config overrides:

```toml
[base_config]
corpus = "MSP-PODCAST"
upstream_model = "wavlm-base-plus"
epoch = 30
batch_size = 16
learning_rate = 1e-5

[[experiments]]
name = "my-experiment"
model_type = "UpstreamFinetune"
config = "base_config"
```

## Project Structure

```
net/                  # Model architecture (wrapper, heads, losses, pooling)
dataset_module/       # SERDataset — loads audio and labels
experiment_runner/    # Training and evaluation loops
utils/                # Config loading, data preprocessing, HF uploader
configs/              # Default JSON configs for categorical/dimensional tasks
```

## Key Features

- **WavLM fine-tuning** with attentive statistics pooling
- **Class imbalance handling** via `BalancedSoftmaxLoss` or `WeightedCrossEntropy`
- **Hybrid early stopping** — stops on smoothed macro-F1 plateau or >5% loss degradation
- **Mixed precision training** (AMP) enabled by default
- **W&B experiment tracking** with per-epoch metrics (F1, UAR, CCC)
