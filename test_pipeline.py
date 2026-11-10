"""
Pipeline verification and self-test script for DEEPSYNC
"""
import os
import sys
import time

print("=" * 60)
print("DEEPSYNC Multi-Modal Verification Test")
print("=" * 60)

import processor
import visual_branch
import audio_branch
import consistency_fusion

test_video = r"D:\2026\SEM 5\faceless-shorts-skill\faceless-shorts\renders\hold-sneeze-share.mp4"
if not os.path.exists(test_video):
    print("Sample video not found at default location. Searching...")
    for root, dirs, files in os.walk(r"D:\2026\SEM 5"):
        for f in files:
            if f.endswith(".mp4"):
                test_video = os.path.join(root, f)
                break
        if os.path.exists(test_video):
            break

print(f"Using test video: {test_video}")

# 1. Test Processor
t0 = time.time()
print("\n[Test 1/5] Running processor.process_video...")
proc_data = processor.process_video(test_video, max_frames=30)
t1 = time.time()
print(f" -> Processed {len(proc_data['frames'])} frames in {t1 - t0:.2f}s")
print(f" -> Audio path: {proc_data['audio_path']}")
print(f" -> Video duration: {proc_data['duration']:.2f}s, Native size: {proc_data['width']}x{proc_data['height']}")

# 2. Test Visual Branch
print("\n[Test 2/5] Running visual_branch.analyze_visual...")
t0 = time.time()
vis_data = visual_branch.analyze_visual(proc_data['frames'], max_vit_crops=5, batch_size=4)
t1 = time.time()
print(f" -> Visual analysis completed in {t1 - t0:.2f}s")
print(f" -> Visual Score (S_V): {vis_data['visual_score']}")
print(f" -> ViT Score: {vis_data['vit_score']}, Laplacian Edge: {vis_data['edge_score']}")
print(f" -> Device used: {vis_data['device_used']}")

# 3. Test Audio Branch
print("\n[Test 3/5] Running audio_branch.analyze_audio...")
t0 = time.time()
aud_data = audio_branch.analyze_audio(proc_data['audio_path'])
t1 = time.time()
print(f" -> Audio analysis completed in {t1 - t0:.2f}s")
print(f" -> Audio Score (S_A): {aud_data['audio_score']}")
print(f" -> Wav2Vec2: {aud_data['wav2vec2_score']}, MFCC Anomaly: {aud_data['mfcc_anomaly_score']}")
print(f" -> MFCC Var: {aud_data['mfcc_var']}, Delta Var: {aud_data['delta_var']}")

# 4. Test Consistency Branch
print("\n[Test 4/5] Running consistency_fusion.analyze_consistency...")
t0 = time.time()
sync_data = consistency_fusion.analyze_consistency(
    proc_data['frames'], proc_data['timestamps'], proc_data['audio_path'], fps=proc_data['fps']
)
t1 = time.time()
print(f" -> Lip-Sync analysis completed in {t1 - t0:.2f}s")
print(f" -> Sync Score (S_C): {sync_data['sync_score']}")
print(f" -> Peak Correlation: {sync_data['peak_correlation']}, Lag: {sync_data['detected_lag_ms']} ms")
print(f" -> Landmark snapshot length: {len(sync_data['landmark_snapshot'])} chars")

# 5. Test Asymmetric Veto Fusion
print("\n[Test 5/5] Running consistency_fusion.fuse_scores...")
fuse_data = consistency_fusion.fuse_scores(
    vis_data['visual_score'], aud_data['audio_score'], sync_data['sync_score']
)
print(f" -> Final Veto Score (S_final): {fuse_data['final_score']}")
print(f" -> Naive Baseline Avg (S_avg): {fuse_data['naive_avg']}")
print(f" -> Verdict: {fuse_data['verdict']}")
print(f" -> Attribution: {fuse_data['attribution']}")
print(f" -> Dilution Trap Neutralized: {fuse_data['dilution_trap_active']}")

# Clean up wav
if os.path.exists(proc_data['audio_path']):
    try:
        os.remove(proc_data['audio_path'])
    except Exception:
        pass

print("\n" + "=" * 60)
print("ALL 5 STAGES TESTED SUCCESSFULLY!")
print("=" * 60)

