import os
import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download, HfApi
from dataset_module.ser_dataset import SERDataset
from net.ser_model_wrapper import SERModel
import utils
from experiment_runner.train import get_class_distribution, identify_head_mid_tail, calculate_group_metrics

def run_evaluate(model_type, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = kwargs.get('batch_size', 32)

    # Dataset Paths Resolution
    wav_dir = kwargs.get('wav_dir', None)
    label_path = kwargs.get('label_path', None)
    corpus = kwargs.get('corpus', None)

    if (wav_dir is None or label_path is None) and corpus and corpus in utils.config:
        corpus_config = utils.config[corpus]
        if wav_dir is None:
            wav_dir = corpus_config.get('PATH_TO_AUDIO')
        if label_path is None:
            label_key = kwargs.get('label_key', 'PATH_TO_LABEL')
            label_path = corpus_config.get(label_key)

    if wav_dir is None or label_path is None:
        print(f"Warning: Could not resolve dataset paths for corpus '{corpus}'")

    # Init WandB
    project = kwargs.get('project', 'SER_Experiment')
    wandb.login(key=utils.WANDB_TOKEN)
    wandb.init(project=project, config=kwargs, reinit=True, tags=kwargs.get('tags', []))

    # Load Group Definitions
    head, mid, tail = identify_head_mid_tail(corpus)

    # Dataset
    split = kwargs.get('split', 'test')
    norm_path = kwargs.get('norm_path', 'stat/norm.pkl')
    wav_mean, wav_std = None, None
    if os.path.exists(norm_path):
        from utils.dataset.dataset import load_norm_stat
        wav_mean, wav_std = load_norm_stat(norm_path)

    eval_dataset = SERDataset(wav_dir, label_path, split=split, wav_mean=wav_mean, wav_std=wav_std)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=SERDataset.collate_fn, num_workers=4)

    # Load model from HF Hub
    hf_repo_id = kwargs.get('hf_repo_id', None)
    if hf_repo_id is None:
        username = HfApi().whoami()["name"]
        corpus_name = (corpus or 'MSP-PODCAST').replace('/', '_')
        upstream = kwargs.get('upstream_model', 'wavlm-base-plus').replace('/', '_')
        loss = kwargs.get('loss_type', 'WeightedCrossEntropy')
        exp_name = kwargs.get('name', None)
        repo_name = f"{corpus_name}_{upstream}_{loss}"
        if exp_name:
            repo_name += f"_{exp_name}"
        hf_repo_id = f"{username}/{repo_name}"

    print(f"Loading model from HF Hub: {hf_repo_id}")
    local_model_dir = snapshot_download(repo_id=hf_repo_id)
    model = SERModel.from_pretrained(local_model_dir)

    model.to(device)
    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating [{split}]"):
            x, y, mask, _ = batch
            x = x.to(device)
            mask = mask.to(device)
            y_indices = torch.argmax(y, dim=1)

            logits = model(x, attention_mask=mask)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = y_indices.cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(targets)

    # Metrics
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average='macro')
    val_uar = recall_score(val_targets, val_preds, average='macro')

    group_metrics = calculate_group_metrics(val_targets, val_preds, head, mid, tail)

    print("="*30)
    print(f"Evaluation on {split} set")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Macro F1: {val_f1:.4f}")
    print(f"UAR:      {val_uar:.4f}")
    print("-" * 20)
    for k, v in group_metrics.items():
        print(f"{k}: {v:.4f}")
    print("="*30)

    wandb.log({
        "test/acc": val_acc,
        "test/f1": val_f1,
        "test/uar": val_uar,
        **{f"test/{k}": v for k, v in group_metrics.items()},
    })
    wandb.finish()

    if 'store_path' in kwargs:
        with open(kwargs['store_path'], 'w') as f:
            f.write(f"Acc: {val_acc}\nF1: {val_f1}\nUAR: {val_uar}\n")
