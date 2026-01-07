
import torch
import numpy as np
from net.loss_modules import get_loss_module

def test_loss_modules():
    # Mock data
    class_counts = {0: 100, 1: 200, 2: 50}
    device = 'cpu' # Use CPU for simple test
    batch_size = 4
    num_classes = 3
    
    # Dummy logits and labels
    logits = torch.randn(batch_size, num_classes)
    labels = torch.tensor([0, 1, 2, 0])
    
    print("Testing SoftmaxLoss...")
    criterion = get_loss_module("Softmax", class_counts, device)
    loss = criterion(logits, labels)
    print(f"Softmax Loss: {loss.item()}")
    
    print("Testing FocalLoss...")
    criterion = get_loss_module("Focal", class_counts, device)
    loss = criterion(logits, labels)
    print(f"Focal Loss: {loss.item()}")

    print("Success!")

if __name__ == "__main__":
    test_loss_modules()
