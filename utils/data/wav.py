import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# Load audio
def extract_wav(wav_path):
    # Use soundfile for fast loading
    wav, sr = sf.read(wav_path)
    
    # ensure float32
    wav = wav.astype(np.float32)

    # Convert to mono if multi-channel (N, C) -> (C, N) or just (N,)
    if wav.ndim > 1:
        # soundfile returns (samples, channels), we want to avg across channels
        wav = np.mean(wav, axis=1)
        
    if sr != 16000:
        # Only resample if necessary (slow, but soundfile covers the fast path)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        
    return wav
def load_audio(audio_path, utts, nj=24):
    # Audio path: directory of audio files
    # utts: list of utterance names with .wav extension
    wav_paths = [os.path.join(audio_path, utt) for utt in utts]
    with Pool(nj) as p:
        wavs = list(tqdm(p.imap(extract_wav, wav_paths), total=len(wav_paths)))
    return wavs