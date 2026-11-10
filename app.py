"""
Stage 6: FastAPI Backend for DEEPSYNC
app.py

Serves the forensic web interface (index.html) and exposes the POST /analyze endpoint
which coordinates the 5-stage multimodal deepfake detection pipeline:
1. processor.py (Demuxing & 15 FPS / 16 kHz sampling)
2. visual_branch.py (MediaPipe Cropping, ViT, Laplacian Edge Ratio)
3. audio_branch.py (Wav2Vec2 spoof classifier, 13 MFCCs, Delta-MFCC, Flatness, ZCR)
4. consistency_fusion.py (6-point Inner-Lip MAR, RMS Energy, VAD Lag-Compensated Cross-Correlation)
5. consistency_fusion.py (Context-Aware Asymmetric Veto Late Fusion)
"""

import os
import shutil
import tempfile
import uuid
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import processor
import visual_branch
import audio_branch
import consistency_fusion

app = FastAPI(
    title="DEEPSYNC Multimodal Deepfake Forensic System",
    description="Multimodal detection combining Vision Transformer, Wav2Vec2, 6-Point Inner-Lip MAR, and Asymmetric Veto Fusion.",
    version="1.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")


@app.get("/")
async def get_index():
    """Serves the DEEPSYNC dark-mode forensic dashboard with cache-busting headers."""
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(
            INDEX_HTML_PATH,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    raise HTTPException(status_code=404, detail="index.html not found.")


@app.get("/Figure_4_1_DEEPSYNC_Stage_Architecture.png")
async def get_architecture_figure():
    """Serves the clean minimalist Figure 4.1 architecture image for reports."""
    fig_path = os.path.join(BASE_DIR, "Figure_4_1_DEEPSYNC_Stage_Architecture.png")
    if os.path.exists(fig_path):
        return FileResponse(
            fig_path,
            media_type="image/png",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    raise HTTPException(status_code=404, detail="Figure 4.1 not found.")


@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """
    Accepts video upload, runs 5-stage multimodal deepfake analysis,
    and returns comprehensive forensic telemetry.
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file submitted.")

    # Create isolated temp folder for this request
    req_id = uuid.uuid4().hex[:8]
    temp_dir = tempfile.mkdtemp(prefix=f"deepsync_{req_id}_")

    try:
        # Save uploaded video
        ext = os.path.splitext(video.filename)[1] or ".mp4"
        temp_video_path = os.path.join(temp_dir, f"input_{req_id}{ext}")
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Stage 1: Demuxing & 15 FPS / 16 kHz Pre-Processing
        proc_data = processor.process_video(
            video_path=temp_video_path,
            output_dir=temp_dir,
            target_fps=15.0,
            target_sr=16000,
            max_frames=150
        )

        frames = proc_data["frames"]
        timestamps = proc_data["timestamps"]
        audio_path = proc_data["audio_path"]
        fps = proc_data["fps"]
        sr = proc_data["sr"]

        # Stage 2: Dual-Stage Visual Forensics (ViT + 2D Laplacian Jawline Ratio)
        visual_results = visual_branch.analyze_visual(
            frames=frames,
            max_vit_crops=30,
            batch_size=8
        )

        # Stage 3: Acoustic Forensics (Wav2Vec2 + 13 MFCC Anomaly)
        audio_results = audio_branch.analyze_audio(
            audio_path=audio_path,
            sr=sr,
            has_audio=proc_data["has_audio"]
        )

        # Stage 4: Lip-Sync Consistency (6-point Inner-Lip MAR vs Acoustic RMS with VAD Lag Search)
        consistency_results = consistency_fusion.analyze_consistency(
            frames=frames,
            timestamps=timestamps,
            audio_path=audio_path,
            fps=fps,
            sr=sr,
            has_audio=proc_data["has_audio"]
        )

        # Stage 5: Context-Aware Asymmetric Veto Late Fusion
        fusion_results = consistency_fusion.fuse_scores(
            visual_score=visual_results["visual_score"],
            audio_score=audio_results["audio_score"],
            sync_score=consistency_results["sync_score"],
            has_audio=proc_data["has_audio"]
        )

        # Assembled Forensic Telemetry Response
        response_payload = {
            "status": "success",
            "filename": video.filename,
            "metadata": {
                "duration_seconds": round(proc_data["duration"], 2),
                "fps": proc_data["fps"],
                "width": proc_data["width"],
                "height": proc_data["height"],
                "has_audio": proc_data["has_audio"],
                "frames_analyzed": len(frames)
            },
            "visual": visual_results,
            "audio": audio_results,
            "consistency": consistency_results,
            "fusion": fusion_results
        }

        return JSONResponse(content=response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("Starting DEEPSYNC Forensic Server at http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

