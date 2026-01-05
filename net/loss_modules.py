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

def get_loss_module(loss_type, class_counts, device='cuda'):
    if loss_type == "WeightedCrossEntropy":
        return WeightedResampledCrossEntropyLoss(class_counts, device)
    elif loss_type == "BalancedSoftmax":
        return BalancedSoftmaxLoss(class_counts, device)
    else:
        # Default to simple CE if unknown
        return nn.CrossEntropyLoss()
