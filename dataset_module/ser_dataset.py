import os
import torch
import pickle as pk
from torch.utils.data import Dataset, DataLoader
from utils.data.podcast import load_cat_emo_label
from utils.data.wav import extract_wav # load_audio unused
from utils.dataset.dataset import WavSet, CAT_EmoSet

class SERDataset(Dataset):
    def __init__(self, audio_path, label_path, split="train", wav_mean=None, wav_std=None, save_norm_path=None):
        """
        Args:
            audio_path (str): Path to audio directory.
            label_path (str): Path to label CSV file.
            split (str): 'train', 'dev', or 'test'.
            wav_mean (float, optional): Mean for normalization.
            wav_std (float, optional): Std for normalization.
            save_norm_path (str, optional): Path to save normalization stats (only for train).
        """
        self.split = split
        self.audio_path = audio_path
        self.label_path = label_path
        
        # Load labels and audio paths
        self.filenames, self.labels = load_cat_emo_label(label_path, split)
        
        # Lazy Loading: Construct full paths instead of loading audio data
        # self.wav_paths = load_audio(audio_path, self.filenames) # Removed eager loading
        self.wav_paths = [os.path.join(audio_path, f) for f in self.filenames]
        
        # Try to load existing stats if available and not provided
        if split == "train" and save_norm_path and os.path.exists(save_norm_path) and (wav_mean is None or wav_std is None):
            from utils.dataset.dataset import load_norm_stat
            print(f"Loading normalization stats from {save_norm_path}")
            wav_mean, wav_std = load_norm_stat(save_norm_path)

        # Initialize sub-datasets
        # Pass paths (self.wav_paths) to WavSet, which now supports lazy loading
        self.wav_set = WavSet(self.wav_paths, wav_mean=wav_mean, wav_std=wav_std)
        self.emo_set = CAT_EmoSet(self.labels)
        
        # Save stats if training and they were computed (not loaded)
        # Note: WavSet computes them if not provided. We should check if they need saving.
        if split == "train" and save_norm_path and not os.path.exists(save_norm_path):
            os.makedirs(os.path.dirname(save_norm_path), exist_ok=True)
            self.wav_set.save_norm_stat(save_norm_path)

    def get_norm_stats(self):
        return self.wav_set.wav_mean, self.wav_set.wav_std

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # WavSet returns (wav, dur)
        audio, dur = self.wav_set[idx]
        # EmoSet returns label
        label = self.emo_set[idx]
        filename = self.filenames[idx]
        
        # Structure compatible with collate_fn_wav_lab_mask which expects [ (wav, dur), lab, utt ]
        # Ensure audio is tensor if needed, but WavSet returns numpy usually, collate handles conversion
        return (audio, dur), label, filename

    @staticmethod
    def collate_fn(batch):
        """
        Wrapper for the legacy collate_fn_wav_lab_mask logic.
        Batch is list of ((wav, dur), label, filename)
        """
        # Legacy collate expects list of [ (wav, dur), lab, utt ]
        # Our __getitem__ returns exactly this tuple.
        # But we need to ensure types match.
        
        # Re-implementing logic here to be self-contained and OO
        # Or import from utils.dataset.collate_fn
        from utils.dataset.collate_fn import collate_fn_wav_lab_mask
        return collate_fn_wav_lab_mask(batch)
