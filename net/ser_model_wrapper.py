import torch
import torch.nn as nn
from transformers import AutoModel
from net.pooling import AttentiveStatisticsPooling
from net.ser import EmotionRegression

class SERModel(nn.Module):
    def __init__(self, ssl_type, pooling_type, head_dim, hidden_dim, classifier_output_dim, dropout=0.2, finetune_layers=3):
        super(SERModel, self).__init__()
        
        # 1. SSL Model
        try:
            self.ssl_model = AutoModel.from_pretrained(ssl_type)
        except OSError:
            # If local path fails or not found, try generic load or raise
            print(f"Warning: Could not load from {ssl_type}, trying default or checking path...")
            raise
            
        self.ssl_model.freeze_feature_encoder()
        
        # Partial Fine-tuning
        self._freeze_layers(finetune_layers)

        # 2. Pooling
        feat_dim = self.ssl_model.config.hidden_size
        if pooling_type == "AttentiveStatisticsPooling":
            self.pool_model = AttentiveStatisticsPooling(feat_dim)
            self.dh_input_dim = feat_dim * 2
        else:
            # Fallback or assume other pooling types take no args or similar
            # If MeanPooling existed, it would be initialized here.
            # For now, relying on explicit support for AttentiveStatisticsPooling
            try:
                from net.pooling import MeanPooling
                if pooling_type == "MeanPooling":
                    self.pool_model = MeanPooling()
                    self.dh_input_dim = feat_dim
                else:
                    raise ValueError(f"Unsupported pooling type: {pooling_type}")
            except ImportError:
                 if pooling_type != "AttentiveStatisticsPooling":
                     raise ValueError(f"Unsupported pooling type: {pooling_type}")

        # 3. Emotion Regression (Classification Head)
        # Note: EmotionRegression init args match net/ser.py: input_dim, hidden_dim, num_layers, output_dim
        # Hardcoding num_layers=1 based on train_cat_ser.py usage (1, 8) 
        # train_cat_ser: EmotionRegression(dh_input_dim, args.head_dim, 1, 8, dropout=0.5)
        # So num_layers is 1. output_dim is 8 (classes).
        self.ser_model = EmotionRegression(self.dh_input_dim, head_dim, 1, classifier_output_dim, dropout=dropout)

    def _freeze_layers(self, finetune_layers):
        # Freeze all encoder layers
        if hasattr(self.ssl_model, "encoder") and hasattr(self.ssl_model.encoder, "layers"):
            for param in self.ssl_model.encoder.layers.parameters():
                param.requires_grad = False
            
            # Unfreeze last N layers
            if finetune_layers > 0:
                for layer in self.ssl_model.encoder.layers[-finetune_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            print("Warning: SSL model structure does not match expected (encoder.layers). Skipping partial freeze.")

    def forward(self, x, attention_mask=None):
        # SSL Forward
        ssl_out = self.ssl_model(x, attention_mask=attention_mask).last_hidden_state
        
        # Pooling
        # AttentiveStatisticsPooling expects (x, mask)
        pool_out = self.pool_model(ssl_out, attention_mask)
        
        # Classification Head
        logits = self.ser_model(pool_out)
        return logits
    
    def save_pretrained(self, save_directory):
        import os
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Save config if needed, or rely on reconstruction
