"""
Stage 1: Demuxing & 15 FPS / 16 kHz Nyquist Pre-Processing
processor.py

Extracts 16 kHz mono PCM WAV and 15 FPS RGB video frames.
Includes robust speech activity checking so silent/muted videos don't cause false positives.
"""

import os
import cv2
import numpy as np
import soundfile as sf
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip


def process_video(video_path: str, output_dir: str = None, target_fps: float = 15.0, target_sr: int = 16000, max_frames: int = 150):
    """
    Demuxes video container into 16 kHz mono WAV and 15 FPS RGB frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(video_path))
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}_16k_mono.wav")

    # --- 1. Audio Demuxing (16 kHz Mono PCM WAV) ---
    has_audio = False
    duration = 0.0
    try:
        clip = VideoFileClip(video_path)
        duration = float(clip.duration) if clip.duration else 0.0
        if clip.audio is not None:
            clip.audio.write_audiofile(
                audio_path,
                fps=target_sr,
                nbytes=2,
                codec='pcm_s16le',
                ffmpeg_params=["-ac", "1"],
                logger=None
            )
            # Verify extracted audio contains actual energy
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                y_check, _ = sf.read(audio_path)
                max_amp = float(np.max(np.abs(y_check))) if len(y_check) > 0 else 0.0
                if max_amp > 1e-3:
                    has_audio = True
                else:
                    has_audio = False
        clip.close()
    except Exception as e:
        print(f"[processor.py] Audio extraction note: {e}")

    # Fallback if no audio track exists
    if not has_audio:
        dummy_duration = max(1.0, duration if duration > 0 else 3.0)
        silence = np.zeros(int(dummy_duration * target_sr), dtype=np.float32)
        sf.write(audio_path, silence, target_sr, subtype='PCM_16')
        has_audio = False

    # --- 2. Temporal Sampling of Video at 15 FPS (RGB) ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file via OpenCV: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if duration <= 0:
        duration = total_video_frames / native_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame timestamps at 15 FPS
    target_interval = 1.0 / target_fps
    sample_timestamps = []
    t = 0.0
    while t < duration and len(sample_timestamps) < max_frames:
        sample_timestamps.append(t)
        t += target_interval

    frames = []
    actual_timestamps = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for target_t in sample_timestamps:
        target_frame_num = int(round(target_t * native_fps))
        if target_frame_num >= total_video_frames and total_video_frames > 0:
            target_frame_num = total_video_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        actual_timestamps.append(target_t)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames could be extracted from video.")

    return {
        "audio_path": audio_path,
        "frames": frames,
        "timestamps": actual_timestamps,
        "fps": target_fps,
        "sr": target_sr,
        "duration": duration,
        "width": width,
        "height": height,
        "has_audio": has_audio
    }
