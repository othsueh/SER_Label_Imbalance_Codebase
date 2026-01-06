import os
import torch
import torch.optim as optim
import wandb
import numpy as np
import time
from tqdm.auto import tqdm
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, recall_score
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from dataset_module.ser_dataset import SERDataset
from net.ser_model_wrapper import SERModel
from net.loss_modules import get_loss_module
import utils

def get_class_distribution(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    dist = {k: v/total for k, v in counts.items()}
    return counts, dist

def identify_head_mid_tail(dist):
    head = [k for k, v in dist.items() if v > 0.10]
    mid = [k for k, v in dist.items() if 0.05 <= v <= 0.10]
    tail = [k for k, v in dist.items() if v < 0.05]
    return head, mid, tail

def calculate_group_metrics(true_labels, pred_labels, head, mid, tail):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    metrics = {}
    
    for group_name, group_classes in [("head", head), ("mid", mid), ("tail", tail)]:
        if not group_classes:
            metrics[f"{group_name}_acc"] = 0.0
            continue
            
        mask = np.isin(true_labels, group_classes)
        if np.sum(mask) == 0:
            metrics[f"{group_name}_acc"] = 0.0
        else:
            group_true = true_labels[mask]
            group_pred = pred_labels[mask]
            metrics[f"{group_name}_acc"] = accuracy_score(group_true, group_pred)
            
    return metrics

def run_train(model_type, **kwargs):
    # Config
    project = kwargs.get('project', 'SER_Experiment')
    seed = kwargs.get('seed', 42)
    epochs = kwargs.get('epoch', 20) # 'epoch' in config, 'epochs' in logic
    batch_size = kwargs.get('batch_size', 32)
    learning_rate = kwargs.get('learning_rate', 1e-4) # 'learning_rate' in config
    patience = kwargs.get('patience', 5)
    loss_type = kwargs.get('loss_type', 'WeightedCrossEntropy')
    use_amp = kwargs.get('use_amp', True)
    gradient_checkpointing = kwargs.get('gradient_checkpointing', False) # Disabled by default due to OOM
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init WandB
    wandb.login(key=utils.WANDB_TOKEN)
    wandb.init(project=project, config=kwargs, reinit=True, tags=kwargs.get('tags', []))
    
    # Dataset Paths Resolution
    wav_dir = kwargs.get('wav_dir', None)
    label_path = kwargs.get('label_path', None)
    
    # Look up from utils.config based on corpus provided in kwargs
    corpus = kwargs.get('corpus', None)
    if (wav_dir is None or label_path is None) and corpus and corpus in utils.config:
        corpus_config = utils.config[corpus]
        if wav_dir is None:
            # config.toml uses 'PATH_TO_AUDIO' or 'PATH_TO_DATASET'? 
            # User config.toml: PATH_TO_AUDIO = '...' PATH_TO_DATASET = '...'
            # train_cat_ser.py used audio_path = config["wav_dir"] -> mapped from config_cat.json
            # experiment.py used dataset = MSPPodcast(config[corpus]['PATH_TO_DATASET'])
            # But here we are using SERDataset which takes wav_dir and label_path explicitly.
            # Let's map PATH_TO_AUDIO.
            wav_dir = corpus_config.get('PATH_TO_AUDIO')
        if label_path is None:
            label_path = corpus_config.get('PATH_TO_LABEL')
            
    if wav_dir is None or label_path is None:
        raise ValueError(f"Could not resolve dataset paths for corpus '{corpus}'. Ensure 'wav_dir'/'label_path' are passed or defined in config.toml under '[{corpus}]'.")
    
    train_dataset = SERDataset(wav_dir, label_path, split="train", save_norm_path=kwargs.get('norm_path', 'ckpt/norm.pkl'))
    # Get Norm Stats for Dev
    wav_mean, wav_std = train_dataset.get_norm_stats()
    val_dataset = SERDataset(wav_dir, label_path, split="dev", wav_mean=wav_mean, wav_std=wav_std)

    print("Preparing the data loader")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=SERDataset.collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=SERDataset.collate_fn, num_workers=4)
    
    # Class Distribution & Groups
    # Convert labels to int for counting if they are not
    # Use existing encoding? utils.data.podcast.load_cat_emo_label returns cur_labs as numpy array of one-hot or int?
    # Inspecting podcast.py: cur_labs = cur_df[["Angry", ...]].to_numpy() -> This is One-Hot-like (N, 8) or probabilities?
    # train_cat_ser.py: y = y.max(dim=1)[1] -> so it is One-Hot or prob distribution.
    # We need integer labels for counting.
    train_labels_int = np.argmax(train_dataset.labels, axis=1)
    class_counts, class_dist = get_class_distribution(train_labels_int)
    head, mid, tail = identify_head_mid_tail(class_dist)
    
    print(f"Memory before model load: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
    print("Preparing the model")
    # Model
    ssl_type = kwargs.get('upstream_model', 'wavlm-large')
    path_to_pretrained = utils.config.get('PATH_TO_PRETRAINED_MODELS')
    if path_to_pretrained:
        local_model_path = os.path.join(path_to_pretrained, ssl_type)
        if os.path.exists(local_model_path):
            print(f"Loading pretrained model from local path: {local_model_path}")
            ssl_type = local_model_path
        else:
             print(f"Local model path not found: {local_model_path}. Fallback to HuggingFace or default.")

    model = SERModel(
        ssl_type=ssl_type, # Should be generic or from config
        pooling_type=kwargs.get('pooling_type', 'AttentiveStatisticsPooling'),
        head_dim=kwargs.get('head_dim', 1024),
        hidden_dim=kwargs.get('hidden_dim', 1024), # Check EmotionRegression init
        classifier_output_dim=len(class_counts),
        dropout=kwargs.get('dropout', 0.2),
        finetune_layers=kwargs.get('finetune_layers', 3)
    )
    
    if gradient_checkpointing:
        print("Enabling Gradient Checkpointing for SSL Model")
        if hasattr(model.ssl_model, "gradient_checkpointing_enable"):
            model.ssl_model.gradient_checkpointing_enable()
        else:
            print("Warning: Model does not support gradient_checkpointing_enable")
            
    model.to(device)
    
    # Loss
    criterion = get_loss_module(loss_type, class_counts, device)
    
    # --- Debug: Model Parameter Status ---
    print("\n" + "="*40)
    print("Checking Model Parameter Status...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print(f"Frozen Parameters:    {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"Memory after model load: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
    
    # Check specific SSL layers
    if hasattr(model, 'ssl_model') and hasattr(model.ssl_model, 'encoder'):
        print("\nChecking SSL Encoder Layer Status:")
        layers = model.ssl_model.encoder.layers
        print(f"Total SSL Layers: {len(layers)}")
        
        # Check first layer (should be frozen)
        first_layer_grad = any(p.requires_grad for p in layers[0].parameters())
        print(f"Layer 0 (Bottom): {'TRAINABLE' if first_layer_grad else 'FROZEN'}")
        
        # Check last layer (should be trainable if finetune > 0)
        last_layer_grad = any(p.requires_grad for p in layers[-1].parameters())
        print(f"Layer {len(layers)-1} (Top):    {'TRAINABLE' if last_layer_grad else 'FROZEN'}")
        
        # Check cut-off point
        finetune_n = kwargs.get('finetune_layers', 3)
        if 0 < finetune_n < len(layers):
            boundary_layer = layers[-(finetune_n + 1)]
            boundary_grad = any(p.requires_grad for p in boundary_layer.parameters())
            print(f"Layer {len(layers)-(finetune_n+1)} (Boundary): {'TRAINABLE' if boundary_grad else 'FROZEN'} (Expected FROZEN)")
    print("="*40 + "\n")
    
    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params
    }, allow_val_change=True)
    # -------------------------------------
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=use_amp)
    
    # Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_targets = []
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        iter_start_time = time.time()
        for i, batch in enumerate(train_pbar):
            # batch: (wav, dur), label, uttid
            # collate_fn returns: total_wav, total_lab, attention_mask, total_utt
            data_start_time = time.time()
            data_time = data_start_time - iter_start_time
            
            x, y, mask, _ = batch
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            # Debug: check graph
            # if epoch == 0 and i == 0:
            #    print(f"x requires_grad: {x.requires_grad}")

            
            # y in dataset is probabilities/one-hot. Criterion expects class indices for CE usually, 
            # unless using SoftLabel CE. PyTorch CE expects indices.
            # Convert y to indices
            y_indices = torch.argmax(y, dim=1)
            
            optimizer.zero_grad()
            
            if i == 0 and epoch == 0:
                print(f"Mem before forward: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

            with autocast(enabled=use_amp):
                logits = model(x, attention_mask=mask)
                loss = criterion(logits, y_indices) # WeightedResampledCrossEntropyLoss wraps CE, expects indices
            
            if i == 0 and epoch == 0:
                print(f"Mem after forward: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

            scaler.scale(loss).backward()
            
            if i == 0 and epoch == 0:
                print(f"Mem after backward: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

            scaler.step(optimizer)
            scaler.update()
            
            compute_time = time.time() - data_start_time
            iter_start_time = time.time() # Reset for next iter
            
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item(), 'dt': f"{data_time:.2f}s", 'ct': f"{compute_time:.2f}s"})
            
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = y_indices.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(all_targets, all_preds)
        train_f1 = f1_score(all_targets, all_preds, average='macro')
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for batch in val_pbar:
                x, y, mask, _ = batch
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                y_indices = torch.argmax(y, dim=1)
                
                logits = model(x, attention_mask=mask)
                loss = criterion(logits, y_indices)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = y_indices.cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_uar = recall_score(val_targets, val_preds, average='macro')
        
        # Group Metrics
        group_metrics = calculate_group_metrics(val_targets, val_preds, head, mid, tail)
        
        # Log
        metrics = {
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/acc": train_acc,
            "train/f1": train_f1,
            "val/loss": avg_val_loss,
            "val/acc": val_acc,
            "val/f1": val_f1,
            "val/uar": val_uar,
            **{f"val/{k}": v for k, v in group_metrics.items()}
        }
        wandb.log(metrics)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save Best Model
            save_path = kwargs.get('save_path', 'ckpt/best_model')
            model.save_pretrained(save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    wandb.finish()
