import torch


def label_adaptive_mixup(x, y, mask, p_mix=0.5):
    """
    Label Adaptive Mixup for Speech Emotion Recognition.

    Mixes pairs of audio samples with a **length-proportional** label blend,
    where the mixing ratio is determined by the actual duration of each sample.

    Audio mixing: fixed 50/50 blend (deterministic)
    Label mixing: length-proportional (adaptive based on audio length)

    Args:
        x:      [B, T] padded waveform (float32, on device)
        y:      [B, C] soft label distributions (float32, on device), sums to 1.0
        mask:   [B, T] binary attention mask (float32, on device), 1=real, 0=pad
        p_mix:  probability of mixing each sample (default 0.5)

    Returns:
        x_mixed:    [B, T] mixed waveforms
        y_mixed:    [B, C] mixed soft labels (still sums to 1.0 per sample)
        mask_mixed: [B, T] union of both masks

    Note:
        - Length derivation: actual_length_i = mask.sum(dim=1)[i]
        - Mix ratio: lambda_i = length_i / (length_i + length_j)
        - Audio: 0.5 * x_i + 0.5 * x_j (fixed, not random)
        - Labels: lambda_i * y_i + (1-lambda_i) * y_j (adaptive)
    """
    B = x.size(0)
    device = x.device

    # Sample which indices to mix (Bernoulli with p_mix)
    mix_flags = torch.bernoulli(torch.full((B,), p_mix, device=device)).bool()  # [B]

    # Random permutation for pairing
    perm = torch.randperm(B, device=device)  # [B]

    # Compute actual audio lengths from attention mask
    # mask is [B, T], 1 for real samples, 0 for padding
    lengths = mask.sum(dim=1)                     # [B], float
    lengths_perm = lengths[perm]                  # [B], float (lengths of paired samples)

    # Length-proportional mixing ratio (per sample)
    # epsilon prevents division by zero (though shouldn't happen with valid data)
    lam = lengths / (lengths + lengths_perm + 1e-6)   # [B], in [0, 1]
    lam = lam.view(B, 1)                               # [B, 1] for broadcasting to [B, C]

    # --- Audio mixing: fixed 50/50 blend ---
    x_mixed = torch.where(
        mix_flags.unsqueeze(1),
        0.5 * x + 0.5 * x[perm],
        x
    )

    # --- Label mixing: length-proportional soft blend ---
    y_mixed = torch.where(
        mix_flags.unsqueeze(1),
        lam * y + (1 - lam) * y[perm],
        y
    )

    # --- Mask mixing: union of both masks ---
    mask_mixed = torch.where(
        mix_flags.unsqueeze(1),
        (mask + mask[perm]).clamp(max=1),
        mask
    )

    return x_mixed, y_mixed, mask_mixed
