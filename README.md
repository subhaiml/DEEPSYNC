# DEEPSYNC: Multimodal Deepfake Forensic Architecture

> **Joint Facial, Voice, and Lip-Sync Spatial-Temporal Analysis with Context-Aware Asymmetric Veto Late Fusion**  
> *Developed by Group-101 | Guide: Prof. Nilanjana Mitra | Department of Computer Science & Engineering (AI & ML), University of Engineering & Management (UEM), Kolkata.*

---

## 📌 Overview

**DEEPSYNC** is an explainable multimodal forensic detection system designed to expose high-realism talking-head deepfakes. Rather than relying on naive single-modality detectors or standard score averaging (which suffers from the **Score Dilution Trap**), DEEPSYNC orchestrates spatial, acoustic, and biomechanical speech dynamics across five distinct analytical stages:

1. **Stage 1: Demuxing & Sampling (`processor.py`)**  
   Demuxes native video containers into 15 FPS sampled video frames (Nyquist limit 7.5 Hz) and 16 kHz mono 16-bit PCM audio (Nyquist limit 8 kHz). Features automatic audio presence detection (`has_audio`) to gracefully handle silent media without false alarms.
2. **Stage 2: Dual-Stage Visual Forensics (`visual_branch.py`)**  
   Primary face detection with dynamic bounding box (+10% margin), deep feature analysis via Vision Transformer (`dima806/deepfake_vs_real_image_detection`), and 2D Laplacian Jawline Boundary Discontinuity ratio ($S_{edge}$) to catch facial feathering and blending boundaries.
3. **Stage 3: Multi-Feature Acoustic Forensics (`audio_branch.py`)**  
   Wav2Vec2 self-supervised spoof classifier (`melodymachine/Deepfake-audio-detection-V2`) combined with 13-band Mel-Frequency Cepstral Coefficients (MFCC) envelope variance, Delta-MFCC velocity variance, Spectral Flatness, and Zero Crossing Rate (ZCR).
4. **Stage 4: Cross-Modal Lip-Sync Consistency (`consistency_fusion.py`)**  
   Scale-invariant 6-point inner-lip Mouth Aspect Ratio (MAR) tracked across 468 MediaPipe FaceMesh landmarks, compared against a 100 ms acoustic RMS energy envelope via VAD-gated lag-compensated Pearson cross-correlation ($\tau \in [-6, +6]$ frames / $\pm 400$ ms).
5. **Stage 5: Context-Aware Asymmetric Veto Late Fusion (`consistency_fusion.py`)**  
   Implements a Dual-Authentic Confidence Gate and Asymmetric Veto Rule ($\alpha = 0.60$):
   $$S_{final} = 0.60 \cdot \max(S_V, S_A, S_C) + 0.40 \cdot \text{Consensus}$$
   Protects against single-modality attacks (such as audio dubs or visual face-swaps) where naive score averaging fails.
6. **Stage 6: Interactive Dashboard & Telemetry (`app.py` & `index.html`)**  
   FastAPI web service paired with a Chart.js forensic interface featuring dual-waveform temporal analysis, explainable 6-row parameter ledger, and one-click print-ready report figure exporter.

---

## 📊 Publication & Report Figures (Print-Ready 300 DPI)

All figures have been rendered with pure white backgrounds (`#FFFFFF`) at 300 DPI for direct insertion into academic reports and presentations:

* **Figure 4.1:** `Figure_4_1_DEEPSYNC_Stage_Architecture.png` (Minimalist six-stage pipeline flowchart)
* **Figure 5.2:** `Figure_5_2_Dual_Waveform_LipSync_Graph.png` (Synchronised Inner-Lip MAR vs Acoustic RMS dynamics)
* **Figure 5.5:** `Figure_5_5_Dilution_Trap_Comparison.png` (Baseline Naive Average vs Asymmetric Veto)
* **Figure 5.7:** `Figure_5_7_Signal_Telemetry_Ledger_Table.png` (Explainable forensic telemetry ledger table)

---

## 🚀 Quick Start

### 1. Requirements
* Python 3.10 – 3.12
* PyTorch (cu121 or CPU)
* FastAPI & Uvicorn
* OpenCV & MediaPipe
* Transformers & Librosa

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-multipart opencv-python mediapipe librosa transformers soundfile matplotlib numpy
```

### 2. Running the Forensic Server
Double-click `start_deepsync.bat` or run:
```bash
python app.py
```
Open your browser at:
```
http://127.0.0.1:8000
```

---

## 📄 License
Academic & Research Use Only. Developed at University of Engineering & Management (UEM), Kolkata.
