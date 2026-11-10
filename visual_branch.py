"""
Stage 2: Dual-Stage Visual Forensics — MediaPipe Cropping, ViT & 2D Laplacian Boundary Ratio
visual_branch.py

Combines deep spatial classification (Vision Transformer) with a training-free 
physical statistic: 2D Laplacian Jawline Boundary-Artifact Ratio (Novelty 1).
Fuses into: S_V = 0.80 * S_ViT + 0.20 * S_edge
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
import mediapipe as mp


def get_working_device() -> int:
    """
    Smart GPU kernel verification (Engineering Case Study 1).
    Proves whether the installed PyTorch binary contains an executable kernel
    for the current GPU architecture (e.g. Blackwell sm_120) by executing a live 1x1 Conv2d.
    Returns:
        0 if GPU kernel execution succeeds, -1 (CPU) otherwise.
    """
    if torch.cuda.is_available():
        try:
            conv = torch.nn.Conv2d(1, 1, 1).cuda()
            test_x = torch.randn(1, 1, 4, 4).cuda()
            _ = conv(test_x)
            print("[visual_branch] Live CUDA Conv2d test passed. Using GPU (device=0).")
            return 0
        except Exception as e:
            print(f"[visual_branch] CUDA Conv2d execution failed ({e}). Gracefully falling back to CPU (device=-1).")
            return -1
    return -1


# Global singletons for efficiency
DEVICE = get_working_device()
_VIT_PIPELINE = None


def get_vit_pipeline():
    global _VIT_PIPELINE
    if _VIT_PIPELINE is None:
        print("[visual_branch] Loading Vision Transformer classifier: dima806/deepfake_vs_real_image_detection...")
        _VIT_PIPELINE = pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection",
            device=DEVICE
        )
    return _VIT_PIPELINE


def crop_primary_faces(frames: list, min_confidence: float = 0.5, pad_ratio: float = 0.10) -> tuple:
    """
    Extracts the primary face bounding box with 10% padding margin from each frame.
    
    Args:
        frames: List of RGB numpy array frames.
        min_confidence: MediaPipe face detection threshold.
        pad_ratio: 10% expansion margin to capture the blending boundary.

    Returns:
        (face_crops, crop_indices, bboxes)
    """
    mp_face_detection = mp.solutions.face_detection
    face_crops = []
    crop_indices = []
    bboxes = []

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as detector:
        for idx, frame in enumerate(frames):
            h, w, _ = frame.shape
            results = detector.process(frame)
            if results.detections:
                # Select the largest face detection
                best_det = None
                max_area = 0
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    area = box.width * box.height
                    if area > max_area:
                        max_area = area
                        best_det = box

                if best_det is not None:
                    xmin = best_det.xmin * w
                    ymin = best_det.ymin * h
                    bw = best_det.width * w
                    bh = best_det.height * h

                    # 10% padding margin on each side to preserve jawline/skin blending seam
                    pad_w = pad_ratio * bw
                    pad_h = pad_ratio * bh

                    x1 = max(0, int(xmin - pad_w))
                    y1 = max(0, int(ymin - pad_h))
                    x2 = min(w, int(xmin + bw + pad_w))
                    y2 = min(h, int(ymin + bh + pad_h))

                    if (x2 - x1) > 20 and (y2 - y1) > 20:
                        crop = frame[y1:y2, x1:x2]
                        face_crops.append(crop)
                        crop_indices.append(idx)
                        bboxes.append((x1, y1, x2, y2))
                        continue

            # Fallback if no face detected in frame: center crop
            ch1, ch2 = int(h * 0.15), int(h * 0.85)
            cw1, cw2 = int(w * 0.20), int(w * 0.80)
            face_crops.append(frame[ch1:ch2, cw1:cw2])
            crop_indices.append(idx)
            bboxes.append((cw1, ch1, cw2, ch2))

    return face_crops, crop_indices, bboxes


def compute_laplacian_boundary_anomaly(crop: np.ndarray) -> float:
    """
    Novelty 1: 2D Laplacian Jawline Boundary-Artifact Ratio.
    Computes spatial sharpness discontinuity between inner facial core (112x112)
    and whole face (224x224) to expose Gaussian feathering boundary seams.

    Equation:
        S_edge = clip( 1.1 * |Var(grad^2 I_inner) - Var(grad^2 I_total)| / max(Var(inner), Var(total)), 0.05, 0.95 )
    """
    resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    
    # 2D Laplacian second-derivative operator
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.abs(lap)

    # Inner core: 112x112 excluding outer 25% boundary where jawline seam lies
    inner_core = abs_lap[56:168, 56:168]

    var_inner = float(np.var(inner_core))
    var_total = float(np.var(abs_lap))

    denom = max(var_inner, var_total, 1e-6)
    anomaly = 1.1 * abs(var_inner - var_total) / denom
    s_edge = float(np.clip(anomaly, 0.05, 0.95))
    return s_edge


def analyze_visual(frames: list, max_vit_crops: int = 30, batch_size: int = 8) -> dict:
    """
    Executes Stage 2 visual forensics.
    
    Returns:
        dict: {
            "visual_score": float (S_V in [0, 1]),
            "vit_score": float (S_ViT in [0, 1]),
            "edge_score": float (S_edge in [0, 1]),
            "face_crops_count": int,
            "vit_sampled_count": int,
            "device_used": str
        }
    """
    if not frames:
        return {
            "visual_score": 0.5,
            "vit_score": 0.5,
            "edge_score": 0.5,
            "face_crops_count": 0,
            "vit_sampled_count": 0,
            "device_used": "cpu"
        }

    # 1. Extract face crops with 10% margin
    face_crops, crop_indices, _ = crop_primary_faces(frames)
    total_crops = len(face_crops)

    # 2. Compute Laplacian boundary edge anomaly over all crops
    edge_scores = [compute_laplacian_boundary_anomaly(c) for c in face_crops]
    s_edge = float(np.mean(edge_scores)) if edge_scores else 0.50

    # 3. Deep spatial classification with Vision Transformer (up to 30 crops, evenly sampled)
    vit_pipeline = get_vit_pipeline()
    num_samples = min(max_vit_crops, total_crops)
    sample_indices = np.linspace(0, total_crops - 1, num_samples, dtype=int)
    sampled_crops = [face_crops[i] for i in sample_indices]

    pil_images = [Image.fromarray(c) for c in sampled_crops]
    vit_scores = []

    # Batch inference
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i + batch_size]
        predictions = vit_pipeline(batch)
        for pred in predictions:
            fake_prob = 0.5
            for item in pred:
                lbl = item["label"].lower()
                if lbl in ["fake", "spoof", "manipulated", "deepfake"]:
                    fake_prob = float(item["score"])
                    break
                elif lbl in ["real", "bonafide", "authentic", "genuine"]:
                    fake_prob = 1.0 - float(item["score"])
            vit_scores.append(fake_prob)

    s_vit = float(np.mean(vit_scores)) if vit_scores else 0.50

    # 4. Fused Visual Score: S_V = 0.80 * S_ViT + 0.20 * S_edge
    s_v = float(np.clip(0.80 * s_vit + 0.20 * s_edge, 0.0, 1.0))

    return {
        "visual_score": round(s_v, 4),
        "vit_score": round(s_vit, 4),
        "edge_score": round(s_edge, 4),
        "face_crops_count": total_crops,
        "vit_sampled_count": len(sampled_crops),
        "device_used": "cuda" if DEVICE == 0 else "cpu"
    }
