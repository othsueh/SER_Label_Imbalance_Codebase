import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

class SigmoidDRLoss(nn.Module):
    """
    Distribution-based Ranking Loss using sigmoid activation.
    Minimizes the distance of negative class probabilities while maximizing
    the distance of positive class probabilities.
    """
    def __init__(self, pos_lambda=1, neg_lambda=0.1/math.log(3.5), L=6., tau=4.):
        super(SigmoidDRLoss, self).__init__()
        self.margin = 0.5
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.L = L
        self.tau = tau

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        dtype = targets.dtype
        device = targets.device
        class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)
        t = targets.unsqueeze(1)
        pos_ind = (t == class_range)
        neg_ind = (t != class_range) * (t >= 0)
        pos_prob = logits[pos_ind].sigmoid()
        neg_prob = logits[neg_ind].sigmoid()
        neg_q = F.softmax(neg_prob/self.neg_lambda, dim=0)
        neg_dist = torch.sum(neg_q * neg_prob)
        if pos_prob.numel() > 0:
            pos_q = F.softmax(-pos_prob/self.pos_lambda, dim=0)
            pos_dist = torch.sum(pos_q * pos_prob)
            loss = self.tau*torch.log(1.+torch.exp(self.L*(neg_dist - pos_dist+self.margin)))/self.L
        else:
            loss = self.tau*torch.log(1.+torch.exp(self.L*(neg_dist - 1. + self.margin)))/self.L
        return loss

class DRLoss(nn.Module):
    """
    Wrapper for SigmoidDRLoss to match the API of other loss modules.
    Supports customizable hyperparameters for distribution-based ranking.
    """
    def __init__(self, device='cuda', pos_lambda=1, neg_lambda=0.1/math.log(3.5), L=6., tau=4.):
        super().__init__()
        # class_counts is unused but kept for API consistency
        self.loss_fn = SigmoidDRLoss(
            pos_lambda=pos_lambda,
            neg_lambda=neg_lambda,
            L=L,
            tau=tau
        )

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, device='cuda', alpha=1.0, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.alpha = alpha

    def forward(self, logits, labels):
        # logits: [B, C], labels: [B]
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def get_loss_module(loss_type, class_counts, device='cuda'):
    if loss_type == "WeightedCrossEntropy":
        return WeightedResampledCrossEntropyLoss(class_counts, device)
    elif loss_type == "BalancedSoftmax":
        return BalancedSoftmaxLoss(class_counts, device)
    elif loss_type == "Softmax":
        return SoftmaxLoss(class_counts, device)
    elif loss_type == "Focal":
        return FocalLoss(device)
    elif loss_type == "DR":
        return DRLoss(device)
    else:
        # Default to simple CE if unknown
        return nn.CrossEntropyLoss()
