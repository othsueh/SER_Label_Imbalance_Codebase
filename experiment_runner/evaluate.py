import os
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score
from torch.utils.data import DataLoader
from dataset_module.ser_dataset import SERDataset
from net.ser_model_wrapper import SERModel
import utils
from experiment_runner.train import get_class_distribution, identify_head_mid_tail, calculate_group_metrics

def run_evaluate(model_type, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = kwargs.get('batch_size', 32)
    model_path = kwargs.get('model_path', 'ckpt/best_model')
    
    # Dataset Paths Resolution
    wav_dir = kwargs.get('wav_dir', None)
    label_path = kwargs.get('label_path', None)
    corpus = kwargs.get('corpus', None)
    
    if (wav_dir is None or label_path is None) and corpus and corpus in utils.config:
        corpus_config = utils.config[corpus]
        if wav_dir is None:
            wav_dir = corpus_config.get('PATH_TO_AUDIO')
        if label_path is None:
            label_path = corpus_config.get('PATH_TO_LABEL')
            
    if wav_dir is None or label_path is None:
         # Fallback or raise error? For now print warning and potentially fail later
         print(f"Warning: Could not resolve dataset paths for corpus '{corpus}'")
    
    # Load Group Definitions from Train data (to be consistent with training metrics)
    print("Loading training labels to define Head/Mid/Tail classes...")
    train_dataset = SERDataset(wav_dir, label_path, split="train")
    train_labels_int = np.argmax(train_dataset.labels, axis=1)
    class_counts, class_dist = get_class_distribution(train_labels_int)
    head, mid, tail = identify_head_mid_tail(class_dist)
    
    # Dataset
    # Split can be passed, default to test or dev
    split = kwargs.get('split', 'test')
    # Need norm stats?
    # If using saved model, we might assume model handles it or valid set uses stats.
    # In train.py we used training stats for Dev. For Test we should ideally use training stats too.
    # Assuming train_norm_stat.pkl exists if trained.
    # Logic in utils.dataset tracks this. 
    # SERDataset __init__ loads it? No, SERDataset takes wav_mean/std as args.
    # We should load stats from file if possible.
    # train.py saved it to 'ckpt/norm.pkl' (default in my impl).
    norm_path = kwargs.get('norm_path', 'stat/norm.pkl')
    wav_mean, wav_std = None, None
    if os.path.exists(norm_path):
        from utils.dataset.dataset import load_norm_stat
        wav_mean, wav_std = load_norm_stat(norm_path)
    
    eval_dataset = SERDataset(wav_dir, label_path, split=split, wav_mean=wav_mean, wav_std=wav_std)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=SERDataset.collate_fn, num_workers=4)
    
    # Validating Model Loading Strategy
    use_from_pretrained = False
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        use_from_pretrained = True
        print(f"Detected config.json in {model_path}. Using SERModel.from_pretrained...")
        model = SERModel.from_pretrained(model_path)
    else:
        print("Initializing model from arguments and loading weights...")
        # Model Setup
        model = SERModel(
            ssl_type=kwargs.get('ssl_type', 'wavlm-large'),
            pooling_type=kwargs.get('pooling_type', 'AttentiveStatisticsPooling'),
            head_dim=kwargs.get('head_dim', 1024),
            hidden_dim=kwargs.get('hidden_dim', 1024),
            classifier_output_dim=len(class_counts),
            dropout=kwargs.get('dropout', 0.2), # dropout doesn't matter for eval
            finetune_layers=0 # not finetuning
        )
        
        # Load Weights
        # If model_path is directory (save_pretrained)
        if os.path.isdir(model_path):
            weights_path = os.path.join(model_path, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            # Assume it's a file
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()
    
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch in eval_loader:
            x, y, mask, _ = batch
            x = x.to(device)
            # y is probability/onehot
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
    
    # If output file specified
    if 'store_path' in kwargs:
         with open(kwargs['store_path'], 'w') as f:
             f.write(f"Acc: {val_acc}\nF1: {val_f1}\nUAR: {val_uar}\n")
