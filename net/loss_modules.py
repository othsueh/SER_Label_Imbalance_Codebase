import torch
import torch.nn as nn
import numpy as np

class WeightedResampledCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with class weights, common in SER tasks.
    """
    def __init__(self, class_counts, device='cuda'):
        super().__init__()
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        weights = []
        # sort keys to ensure order 0..N
        sorted_keys = sorted(class_counts.keys())
        # Assuming keys are integers 0 to N-1
        for cls in sorted_keys:
             freq = class_counts[cls]
             w = total_samples / (num_classes * freq) if freq > 0 else 0
             weights.append(w)
        
        self.weights = torch.tensor(weights, device=device, dtype=torch.float)
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, logits, labels):
        return self.criterion(logits, labels)

class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss: https://arxiv.org/abs/2007.10740
    """
    def __init__(self, class_counts, device='cuda'):
        super().__init__()
        self.class_counts = class_counts
        # Convert dict to sorted list/tensor
        sorted_keys = sorted(class_counts.keys())
        counts = [class_counts[k] for k in sorted_keys]
        self.sample_per_class = torch.tensor(counts, device=device, dtype=torch.float)

    def forward(self, logits, labels):
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + torch.log(spc + 1e-9)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

class SoftmaxLoss(nn.Module):
    """
    Standard Cross Entropy Loss
    """
    def __init__(self, class_counts, device='cuda'):
        # class_counts is unused here but kept for API consistency
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.criterion(logits, labels)

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, class_counts, device='cuda', alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.device = device
        
        # Handle alpha
        # If alpha is None, try to balance based on class counts similar to WeightedResampledCE
        if alpha is None:
            total_samples = sum(class_counts.values())
            num_classes = len(class_counts)
            weights = []
            sorted_keys = sorted(class_counts.keys())
            for cls in sorted_keys:
                 freq = class_counts[cls]
                 w = total_samples / (num_classes * freq) if freq > 0 else 0
                 weights.append(w)
            self.alpha = torch.tensor(weights, device=device, dtype=torch.float)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, device=device, dtype=torch.float)
        else:
            self.alpha = alpha # Can be a single float or None if no alpha weighing is desired

    def forward(self, logits, labels):
        # logits: [B, C], labels: [B]
        ce_loss = nn.CrossEntropyLoss(reduction='none', weight=self.alpha)(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def get_loss_module(loss_type, class_counts, device='cuda'):
    if loss_type == "WeightedCrossEntropy":
        return WeightedResampledCrossEntropyLoss(class_counts, device)
    elif loss_type == "BalancedSoftmax":
        return BalancedSoftmaxLoss(class_counts, device)
    elif loss_type == "Softmax":
        return SoftmaxLoss(class_counts, device)
    elif loss_type == "Focal":
        return FocalLoss(class_counts, device)
    else:
        # Default to simple CE if unknown
        return nn.CrossEntropyLoss()
