
import os
import torch
import numpy as np
import soundfile as sf
import tempfile
from utils.dataset.dataset import WavSet

def test_truncation():
    # Create a 60 second dummy wav
    sr = 16000
    duration = 60
    wav = np.random.uniform(-1, 1, int(sr * duration)).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, wav, sr)
        tmp_path = tmp.name
        
    try:
        # Initialize WavSet
        # By default max_dur should be 12 (from code reading)
        dataset = WavSet([tmp_path], wav_mean=0.0, wav_std=1.0)
        
        print(f"Dataset max_dur (samples): {dataset.max_dur}")
        print(f"Dataset max_dur (seconds): {dataset.max_dur / sr}")
        
        # Get item
        item, dur = dataset[0]
        print(f"Original Length: {len(wav)}")
        print(f"Loaded Length: {len(item)}")
        print(f"Duration reported: {dur}")
        
        if len(item) > sr * 13:
             print("FAIL: Item was not truncated correctly!")
        else:
             print("SUCCESS: Item was truncated.")
             
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    test_truncation()
