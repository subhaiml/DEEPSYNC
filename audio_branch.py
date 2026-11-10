"""
Stage 3: Acoustic Forensics — 13 MFCCs, Delta-MFCCs, Spectral Flatness, ZCR & Wav2Vec2
audio_branch.py

Combines deep speech representation classification (Wav2Vec2) with explainable
spectral statistics (13 MFCCs, Delta-MFCCs, Spectral Flatness, Zero-Crossing Rate).
Safely handles silent or muted video files without false-positive penalty.
"""

import numpy as np
import librosa
from transformers import pipeline
from visual_branch import DEVICE

_AUDIO_PIPELINE = None


def get_audio_pipeline():
    global _AUDIO_PIPELINE
    if _AUDIO_PIPELINE is None:
        print("[audio_branch] Loading Wav2Vec2 spoof classifier: melodymachine/Deepfake-audio-detection-V2...")
        _AUDIO_PIPELINE = pipeline(
            "audio-classification",
            model="melodymachine/Deepfake-audio-detection-V2",
            device=DEVICE
        )
    return _AUDIO_PIPELINE


def extract_explainable_features(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Extracts classical, explainable acoustic features:
    - 13 Mel-Frequency Cepstral Coefficients (MFCCs)
    - Mean variance across MFCC bands (low variance indicates neural vocoder oversmoothing)
    - First-order Delta-MFCCs and their variance (vocal tract velocity)
    - Spectral Flatness (tonality vs noise)
    - Zero-Crossing Rate (ZCR)
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

    # Equation 4.5
    spectral_anomaly = float(np.clip(1.0 - mfcc_var / 850.0, 0.08, 0.92))

    delta_mfccs = librosa.feature.delta(mfccs)
    delta_var = float(np.mean(np.var(delta_mfccs, axis=1)))

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    return {
        "mfcc_anomaly_score": round(spectral_anomaly, 4),
        "mfcc_var": round(mfcc_var, 4),
        "delta_var": round(delta_var, 4),
        "spectral_flatness": round(flatness, 6),
        "zcr": round(zcr, 6)
    }


def analyze_audio(audio_path: str, sr: int = 16000, has_audio: bool = True) -> dict:
    """
    Executes Stage 3 acoustic forensics.
    """
    if not has_audio:
        return {
            "audio_score": 0.10,
            "wav2vec2_score": 0.05,
            "mfcc_anomaly_score": 0.10,
            "mfcc_var": 0.0,
            "delta_var": 0.0,
            "spectral_flatness": 0.0,
            "zcr": 0.0,
            "has_audio": False,
            "bypassed": True,
            "status_note": "No audio stream detected (Acoustic analysis bypassed)"
        }

    # Load 16 kHz mono waveform
    y, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    if len(y) == 0 or np.max(np.abs(y)) < 1e-3:
        return {
            "audio_score": 0.10,
            "wav2vec2_score": 0.05,
            "mfcc_anomaly_score": 0.10,
            "mfcc_var": 0.0,
            "delta_var": 0.0,
            "spectral_flatness": 0.0,
            "zcr": 0.0,
            "has_audio": False,
            "bypassed": True,
            "status_note": "Silent audio stream (Acoustic analysis bypassed)"
        }

    # 1. Classical explainable features
    feats = extract_explainable_features(y, sr=sr)
    s_mfcc = feats["mfcc_anomaly_score"]

    # 2. Deep acoustic classification via Wav2Vec2
    audio_pipe = get_audio_pipeline()
    predictions = audio_pipe({"raw": y, "sampling_rate": sr})

    s_wav2vec2 = 0.5
    for item in predictions:
        lbl = item["label"].lower()
        if lbl in ["fake", "spoof", "synthetic", "cloned"]:
            s_wav2vec2 = float(item["score"])
            break
        elif lbl in ["real", "bonafide", "genuine", "authentic"]:
            s_wav2vec2 = 1.0 - float(item["score"])
            break

    # 3. Audio Score Combination: S_A = 0.75 * S_Wav2Vec2 + 0.25 * S_MFCC_anomaly
    s_a = float(np.clip(0.75 * s_wav2vec2 + 0.25 * s_mfcc, 0.0, 1.0))

    return {
        "audio_score": round(s_a, 4),
        "wav2vec2_score": round(s_wav2vec2, 4),
        "mfcc_anomaly_score": feats["mfcc_anomaly_score"],
        "mfcc_var": feats["mfcc_var"],
        "delta_var": feats["delta_var"],
        "spectral_flatness": feats["spectral_flatness"],
        "zcr": feats["zcr"],
        "has_audio": True,
        "bypassed": False,
        "status_note": "Acoustic spoofing analysis complete"
    }
