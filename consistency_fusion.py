"""
Stage 4 & Stage 5: Lip-Sync Consistency & Context-Aware Asymmetric Veto Late Fusion
consistency_fusion.py

Stage 4:
- Scale-Invariant 6-Point Inner-Lip Mouth Aspect Ratio (MAR) via MediaPipe Face Mesh
- Synchronised Acoustic RMS Energy Envelope (100 ms integration window)
- Voice-Activity-Gated, Lag-Compensated Cross-Correlation over [-6, +6] frames (+/-400 ms)

Stage 5:
- Dual-Authentic Confidence Gate
- Context-Aware Asymmetric Veto Late Fusion (alpha = 0.60, weights 0.40 / 0.35 / 0.25)
- Decision threshold tau = 0.55 with failure-mode attribution
"""

import base64
import cv2
import numpy as np
import librosa
from scipy.stats import pearsonr
import mediapipe as mp


def compute_inner_lip_mar(landmarks, width: int, height: int):
    """
    Novelty 2A: Scale-Invariant Six-Point Inner-Lip Mouth Aspect Ratio.
    Vertical pairs: (82, 87), (13, 14), (312, 317)
    Horizontal mouth corners: (78, 308)
    
    MAR = (||P82 - P87|| + ||P13 - P14|| + ||P312 - P317||) / (3 * ||P78 - P308||)
    """
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    p82, p87 = pt(82), pt(87)
    p13, p14 = pt(13), pt(14)
    p312, p317 = pt(312), pt(317)
    p78, p308 = pt(78), pt(308)

    d_vertical = (
        np.linalg.norm(p82 - p87) +
        np.linalg.norm(p13 - p14) +
        np.linalg.norm(p312 - p317)
    ) / 3.0

    d_horizontal = np.linalg.norm(p78 - p308)

    if d_horizontal < 1e-4:
        return 0.0

    return float(d_vertical / d_horizontal)


def generate_landmark_snapshot(frame_rgb: np.ndarray, landmarks) -> str:
    """
    Renders an explainable forensic thumbnail with inner-lip landmarks and axes.
    Returns Base64 encoded JPEG string.
    """
    h, w, _ = frame_rgb.shape
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def pt(idx):
        lm = landmarks[idx]
        return (int(round(lm.x * w)), int(round(lm.y * h)))

    # Landmark indices
    p82, p87 = pt(82), pt(87)
    p13, p14 = pt(13), pt(14)
    p312, p317 = pt(312), pt(317)
    p78, p308 = pt(78), pt(308)

    # Draw horizontal axis (cyan)
    cv2.line(vis, p78, p308, (255, 230, 0), 2, cv2.LINE_AA)

    # Draw vertical axes (rose / magenta)
    for p_top, p_bot in [(p82, p87), (p13, p14), (p312, p317)]:
        cv2.line(vis, p_top, p_bot, (180, 50, 255), 2, cv2.LINE_AA)

    # Draw landmark dots (green)
    for p in [p78, p308, p82, p87, p13, p14, p312, p317]:
        cv2.circle(vis, p, 4, (0, 255, 128), -1, cv2.LINE_AA)
        cv2.circle(vis, p, 5, (0, 0, 0), 1, cv2.LINE_AA)

    # Crop tightly around mouth for crisp visualization
    cx = (p78[0] + p308[0]) // 2
    cy = (p13[1] + p14[1]) // 2
    box_size = int(max(np.linalg.norm(np.array(p78) - np.array(p308)) * 1.5, 140))

    x1 = max(0, cx - box_size // 2)
    y1 = max(0, cy - box_size // 2)
    x2 = min(w, cx + box_size // 2)
    y2 = min(h, cy + box_size // 2)

    mouth_crop = vis[y1:y2, x1:x2]
    if mouth_crop.size == 0:
        mouth_crop = vis

    mouth_crop = cv2.resize(mouth_crop, (280, 280), interpolation=cv2.INTER_AREA)

    # Encode to base64
    _, buf = cv2.imencode(".jpg", mouth_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64_str = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def compute_rms_envelope(y: np.ndarray, timestamps: list, sr: int = 16000, window_duration: float = 0.10) -> np.ndarray:
    """
    Computes acoustic RMS energy over a 100 ms integration window centered on each frame timestamp.
    window_samples = int(0.10 * sr) = 1600 samples
    """
    window_samples = int(window_duration * sr)
    half_win = window_samples // 2
    total_samples = len(y)

    rms_values = []
    for t in timestamps:
        center = int(round(t * sr))
        start = max(0, center - half_win)
        end = min(total_samples, center + half_win)
        segment = y[start:end]

        if len(segment) > 0:
            rms = float(np.sqrt(np.mean(segment ** 2)))
        else:
            rms = 0.0
        rms_values.append(rms)

    return np.array(rms_values, dtype=np.float32)


def smooth_and_normalize(series: np.ndarray) -> np.ndarray:
    """
    3-point moving average smoothing followed by Min-Max normalisation to [0, 1].
    """
    if len(series) < 3:
        return np.zeros_like(series)

    kernel = np.ones(3, dtype=np.float32) / 3.0
    smoothed = np.convolve(series, kernel, mode="same")

    min_val = float(np.min(smoothed))
    max_val = float(np.max(smoothed))
    ptp = max_val - min_val

    if ptp > 1e-6:
        norm = (smoothed - min_val) / ptp
    else:
        norm = np.zeros_like(smoothed)

    return norm


def vad_lag_cross_correlation(mar_norm: np.ndarray, rms_norm: np.ndarray, fps: float = 15.0, max_lag_frames: int = 6) -> tuple:
    """
    Novelty 2B: Voice-Activity-Gated, Lag-Compensated Cross-Correlation.
    Searches lag window tau in [-6, +6] frames (+/-400 ms).
    Excludes silent frames where no audible speech and mouth closed.
    """
    n = len(mar_norm)
    if n < 8:
        return 0.0, 0.0, 0.50

    best_corr = -1.0
    best_lag = 0

    for tau in range(-max_lag_frames, max_lag_frames + 1):
        if tau >= 0:
            m_slice = mar_norm[:n - tau] if tau > 0 else mar_norm
            r_slice = rms_norm[tau:] if tau > 0 else rms_norm
        else:
            m_slice = mar_norm[-tau:]
            r_slice = rms_norm[:n + tau]

        # VAD active speech condition: audible acoustic energy or substantial mouth opening
        active_idx = (r_slice > 0.12) | (m_slice > 0.25)
        num_active = int(np.sum(active_idx))

        if num_active >= 5:
            m_sub = m_slice[active_idx]
            r_sub = r_slice[active_idx]

            std_m = np.std(m_sub)
            std_r = np.std(r_sub)

            if std_m > 1e-4 and std_r > 1e-4:
                r_val, _ = pearsonr(m_sub, r_sub)
                if not np.isnan(r_val) and r_val > best_corr:
                    best_corr = float(r_val)
                    best_lag = tau
            else:
                corr_zero = 0.0
                if corr_zero > best_corr:
                    best_corr = corr_zero
                    best_lag = tau

    if best_corr == -1.0:
        best_corr = 0.0
        best_lag = 0

    detected_lag_ms = float(best_lag * (1000.0 / fps))

    # Map peak correlation monotonically to lip-sync inconsistency score S_C in [0.08, 0.92]
    s_c = float(np.clip(0.92 - 0.94 * best_corr, 0.08, 0.92))

    return best_corr, detected_lag_ms, s_c


def analyze_consistency(frames: list, timestamps: list, audio_path: str, fps: float = 15.0, sr: int = 16000, has_audio: bool = True) -> dict:
    """
    Executes Stage 4 Lip-Sync Consistency.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    h, w, _ = frames[0].shape

    mp_face_mesh = mp.solutions.face_mesh
    mar_raw = []
    middle_idx = len(frames) // 2
    landmark_snapshot_b64 = None

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as mesh:
        for idx, frame in enumerate(frames):
            results = mesh.process(frame)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                mar = compute_inner_lip_mar(lms, w, h)
                mar_raw.append(mar)

                if idx == middle_idx or (landmark_snapshot_b64 is None and idx > middle_idx // 2):
                    landmark_snapshot_b64 = generate_landmark_snapshot(frame, lms)
            else:
                last_val = mar_raw[-1] if mar_raw else 0.20
                mar_raw.append(last_val)

    if landmark_snapshot_b64 is None:
        mid_frame = frames[middle_idx]
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))
        landmark_snapshot_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    mar_array = np.array(mar_raw, dtype=np.float32)
    mar_norm = smooth_and_normalize(mar_array)

    # Check if audio is present and audible
    audio_energy = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    if not has_audio or audio_energy < 1e-3:
        # Video is silent / muted. Lip-sync cannot be measured without speech audio.
        return {
            "sync_score": 0.15,
            "peak_correlation": 0.0,
            "detected_lag_ms": 0.0,
            "mar_series": [round(float(v), 4) for v in mar_norm],
            "rms_series": [0.0 for _ in mar_norm],
            "timestamps": [round(float(t), 3) for t in timestamps],
            "landmark_snapshot": landmark_snapshot_b64,
            "has_audio": False,
            "bypassed": True,
            "status_note": "No audio stream detected (Lip-sync analysis bypassed)"
        }

    rms_array = compute_rms_envelope(y, timestamps, sr=sr)
    rms_norm = smooth_and_normalize(rms_array)

    peak_corr, lag_ms, s_c = vad_lag_cross_correlation(mar_norm, rms_norm, fps=fps)

    return {
        "sync_score": round(s_c, 4),
        "peak_correlation": round(peak_corr, 4),
        "detected_lag_ms": round(lag_ms, 1),
        "mar_series": [round(float(v), 4) for v in mar_norm],
        "rms_series": [round(float(v), 4) for v in rms_norm],
        "timestamps": [round(float(t), 3) for t in timestamps],
        "landmark_snapshot": landmark_snapshot_b64,
        "has_audio": True,
        "bypassed": False,
        "status_note": "Lip-sync cross-correlation complete"
    }


def fuse_scores(visual_score: float, audio_score: float, sync_score: float, has_audio: bool = True) -> dict:
    """
    Stage 5: Context-Aware Asymmetric Veto Late Fusion.
    """
    s_v = float(visual_score)
    s_a = float(audio_score)
    s_c = float(sync_score)

    if not has_audio:
        # Video-only media (silent/no audio stream)
        s_final = s_v
        s_avg = s_v
        is_deepfake = s_final >= 0.55
        verdict = "MANIPULATED / DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC MEDIA"
        attribution = "Visual / Facial Manipulation (Video-Only)" if is_deepfake else "Authentic Visual Media (Video-Only)"

        return {
            "final_score": round(s_final, 4),
            "naive_avg": round(s_avg, 4),
            "sync_eff": 0.15,
            "dual_authentic_gate": False,
            "dilution_trap_active": False,
            "verdict": verdict,
            "is_deepfake": is_deepfake,
            "attribution": attribution,
            "max_modality_score": round(s_v, 4),
            "has_audio": False
        }

    # 1. Dual-Authentic Confidence Gate
    dual_authentic_triggered = False
    if s_v < 0.38 and s_a < 0.38 and s_c < 0.80:
        s_sync_eff = 0.55 * s_c
        dual_authentic_triggered = True
    else:
        s_sync_eff = s_c

    # 2. Asymmetric Veto Fusion (alpha = 0.60)
    max_modality = max(s_v, s_a, s_sync_eff)
    weighted_consensus = (0.40 * s_v) + (0.35 * s_a) + (0.25 * s_sync_eff)
    s_final = (0.60 * max_modality) + (0.40 * weighted_consensus)
    s_final = float(np.clip(s_final, 0.0, 1.0))

    # 3. Naive Average Baseline
    s_avg = float(np.clip((s_v + s_a + s_c) / 3.0, 0.0, 1.0))

    # 4. Verdict & Failure-Mode Attribution
    is_deepfake = s_final >= 0.55
    verdict = "MANIPULATED / DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC MEDIA"

    attribution = "All Modalities Within Authentic Baseline"
    if is_deepfake:
        if max_modality == s_v:
            attribution = "Visual / Facial Manipulation (ViT + Laplacian Boundary)"
        elif max_modality == s_a:
            attribution = "Acoustic / Synthetic Voice Spoof (Wav2Vec2 + MFCC)"
        else:
            attribution = "Audio-Visual Lip-Sync Inconsistency / Re-dubbing (MAR vs RMS)"

    # Check whether the Dilution Trap was neutralized
    dilution_trap_active = (s_final >= 0.55 and s_avg < 0.55)

    return {
        "final_score": round(s_final, 4),
        "naive_avg": round(s_avg, 4),
        "sync_eff": round(s_sync_eff, 4),
        "dual_authentic_gate": dual_authentic_triggered,
        "dilution_trap_active": dilution_trap_active,
        "verdict": verdict,
        "is_deepfake": is_deepfake,
        "attribution": attribution,
        "max_modality_score": round(max_modality, 4),
        "has_audio": True
    }
