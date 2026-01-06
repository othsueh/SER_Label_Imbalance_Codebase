import os
import librosa
from tqdm import tqdm
from multiprocessing import Pool

import torch
import torchaudio

# Load audio
def extract_wav(wav_path):
    # Using torchaudio for faster loading
    wav, sr = torchaudio.load(wav_path)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
        
    # Convert to mono if needed (average across channels)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        
    # Squeeze to (T,) and convert to numpy to match librosa behavior
    return wav.squeeze().numpy()
def load_audio(audio_path, utts, nj=24):
    # Audio path: directory of audio files
    # utts: list of utterance names with .wav extension
    wav_paths = [os.path.join(audio_path, utt) for utt in utts]
    with Pool(nj) as p:
        wavs = list(tqdm(p.imap(extract_wav, wav_paths), total=len(wav_paths)))
    return wavs