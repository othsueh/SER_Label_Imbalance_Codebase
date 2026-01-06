import time
import torch
import torchaudio
import librosa
import numpy as np
import os
from utils.data.wav import extract_wav as fast_extract_wav

def create_dummy_wav(filename, duration=10, sr=16000):
    # Generate 10 seconds of noise
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t) 
    # Librosa writes float32, torchaudio expects tensor
    # Let's use soundfile to write or torchaudio
    torchaudio.save(filename, torch.from_numpy(x).float().unsqueeze(0), sr)
    print(f"Created dummy wav: {filename} ({duration}s, {sr}Hz)")

def benchmark_librosa(filename, n_trials=50):
    start = time.time()
    for _ in range(n_trials):
        _ = librosa.load(filename, sr=16000)
    end = time.time()
    return (end - start) / n_trials

def benchmark_new_loader(filename, n_trials=50):
    start = time.time()
    for _ in range(n_trials):
        _ = fast_extract_wav(filename)
    end = time.time()
    return (end - start) / n_trials

def main():
    dummy_file = "temp_benchmark.wav"
    try:
        create_dummy_wav(dummy_file)
        
        print("\nStarting Benchmark (50 trials)...")
        
        # 1. Librosa
        print("Benchmarking librosa.load (Baseline)...")
        librosa_time = benchmark_librosa(dummy_file)
        print(f"Librosa avg time: {librosa_time*1000:.2f} ms")
        
        # 2. Torchaudio (via utils.data.wav)
        print("Benchmarking utils.data.wav.extract_wav (Optimized)...")
        new_time = benchmark_new_loader(dummy_file)
        print(f"New Loader avg time: {new_time*1000:.2f} ms")
        
        speedup = librosa_time / new_time
        print(f"\nSpeedup: {speedup:.2f}x")
        
    finally:
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

if __name__ == "__main__":
    main()
