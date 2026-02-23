# net/emotion2vec_wrapper.py

import torch
import torch.nn as nn


class _FakeConfig:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size


class _FakeEncoder:
    def __init__(self, layers: nn.ModuleList):
        self.layers = layers


class _ForwardOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class Emotion2VecWrapper(nn.Module):
    """
    Adapts FunASR emotion2vec to the HuggingFace-compatible interface used by SERModel:
      - .config.hidden_size      -> 768
      - .encoder.layers          -> nn.ModuleList of 8 transformer blocks
      - .freeze_feature_encoder() -> freezes CNN local_encoder
      - forward(x, attention_mask=...).last_hidden_state -> (B, T, 768)
    """
    HIDDEN_SIZE = 768  # emotion2vec_plus_base embed_dim

    def __init__(self, model_id: str = "iic/emotion2vec_plus_base"):
        super().__init__()
        try:
            from funasr import AutoModel as FunASRAutoModel
        except ImportError:
            raise ImportError("funasr is required for emotion2vec. Install: pip install funasr")

        print(f"[Emotion2VecWrapper] Loading '{model_id}' via FunASR...")

        # model_id can be either:
        # - HuggingFace model ID: "iic/emotion2vec_plus_base" (downloads from ModelScope)
        # - Local path: "/path/to/emotion2vec_plus_base" (loads from disk)
        funasr_automodel = FunASRAutoModel(model=model_id, device="cpu")
        self._e2v = funasr_automodel.model  # underlying Emotion2vec nn.Module

        self.config = _FakeConfig(hidden_size=self.HIDDEN_SIZE)

        # Expose encoder layers for _freeze_layers() in SERModel
        # Primary path: self._e2v.blocks (nn.ModuleList of 8 transformer blocks)
        if hasattr(self._e2v, "blocks") and isinstance(self._e2v.blocks, nn.ModuleList):
            self.encoder = _FakeEncoder(layers=self._e2v.blocks)
        else:
            # Fallback discovery for future FunASR versions
            for attr in ["encoder", "transformer", "layers"]:
                candidate = getattr(self._e2v, attr, None)
                if isinstance(candidate, nn.ModuleList):
                    print(f"[Emotion2VecWrapper] 'blocks' not found; using '{attr}' as fallback.")
                    self.encoder = _FakeEncoder(layers=candidate)
                    break
            else:
                raise AttributeError(
                    f"Cannot find transformer encoder layers in {type(self._e2v).__name__}. "
                    f"Tried: blocks, encoder, transformer, layers."
                )

    def freeze_feature_encoder(self):
        """Freeze CNN convolutional layers (analogous to WavLM's freeze_feature_encoder)."""
        audio_enc = None
        if hasattr(self._e2v, "modality_encoders"):
            # modality_encoders is a ModuleDict, use bracket notation
            if "AUDIO" in self._e2v.modality_encoders:
                audio_enc = self._e2v.modality_encoders["AUDIO"]

        if audio_enc is not None and hasattr(audio_enc, "local_encoder"):
            target = audio_enc.local_encoder
            label = "modality_encoders['AUDIO'].local_encoder"
        elif audio_enc is not None:
            target = audio_enc
            label = "modality_encoders['AUDIO'] (full AudioEncoder)"
        else:
            print("[Emotion2VecWrapper] Warning: CNN feature encoder not found; skipping freeze.")
            return

        count = 0
        for param in target.parameters():
            param.requires_grad = False
            count += 1
        print(f"[Emotion2VecWrapper] Frozen {count} param tensors in {label}.")

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> _ForwardOutput:
        """
        Args:
            input_values: (B, T_samples) z-normalized 16kHz waveform
            attention_mask: (B, T_samples) binary mask — 1=real, 0=padding (HF convention)
        Returns:
            _ForwardOutput with .last_hidden_state: (B, T_frames, 768)
        """
        # HF convention: attention_mask=1 means real; FunASR padding_mask=True means padding
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)  # invert: (B, T_samples)
        else:
            B, T = input_values.shape
            padding_mask = torch.zeros(B, T, dtype=torch.bool, device=input_values.device)

        # extract_features is differentiable; mask=False disables random masking during training
        # remove_extra_tokens=True strips the 10 prepended learnable tokens
        result = self._e2v.extract_features(
            source=input_values,
            padding_mask=padding_mask,
            mask=False,
            remove_extra_tokens=True,
        )
        return _ForwardOutput(last_hidden_state=result["x"])  # (B, T_frames, 768)
