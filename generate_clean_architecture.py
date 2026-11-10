"""
generate_clean_architecture.py
Generates a simplified, minimalist, high-resolution Figure 4.1 on a pure white background
specifically designed for academic reports and paper printing.
"""

from PIL import Image, ImageDraw, ImageFont
import math
import os

def create_figure_4_1(output_path="Figure_4_1_DEEPSYNC_Stage_Architecture.png"):
    width = 1360
    height = 700
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Fonts
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 24)
        sub_font = ImageFont.truetype("arial.ttf", 14)
        box_title_font = ImageFont.truetype("arialbd.ttf", 15)
        box_module_font = ImageFont.truetype("arialbd.ttf", 12)
        box_body_font = ImageFont.truetype("arial.ttf", 12)
        score_font = ImageFont.truetype("arialbd.ttf", 13)
        badge_font = ImageFont.truetype("arialbd.ttf", 11)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = title_font
        box_title_font = title_font
        box_module_font = title_font
        box_body_font = title_font
        score_font = title_font
        badge_font = title_font

    # --- Title Banner ---
    draw.text((50, 30), "DEEPSYNC: Multimodal Deepfake Forensic Architecture", fill="#0F172A", font=title_font)
    draw.text((50, 62), "Figure 4.1: Six-stage pipeline mapping talking-head video to multimodal forensic verdict", fill="#64748B", font=sub_font)
    draw.line([(50, 88), (width - 50, 88)], fill="#E2E8F0", width=2)

    # Box Drawing Function
    def draw_box(x, y, w, h, title, module_name, bullets, score_str, stroke_color, fill_color, badge=None):
        # Card background & border
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill_color, outline=stroke_color, width=2)

        # Header Title
        draw.text((x + 14, y + 12), title, fill="#0F172A", font=box_title_font)

        # Module Tag (e.g. processor.py)
        if module_name:
            draw.text((x + 14, y + 33), module_name, fill=stroke_color, font=box_module_font)

        # Badge pill
        if badge:
            bw, bh = 72, 20
            bx, by = x + w - bw - 12, y + 10
            draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=4, fill="#FFFFFF", outline=stroke_color, width=1)
            draw.text((bx + 8, by + 3), badge, fill=stroke_color, font=badge_font)

        # Bullets (Max 2 concise lines)
        curr_y = y + 54 if module_name else y + 38
        for b in bullets:
            draw.text((x + 14, curr_y), b, fill="#334155", font=box_body_font)
            curr_y += 18

        # Output score highlight
        if score_str:
            draw.text((x + 14, y + h - 22), score_str, fill="#0F172A", font=score_font)

    # Arrow Drawing Function
    def draw_arrow(p1, p2, color="#64748B", width_line=2):
        x1, y1 = p1
        x2, y2 = p2
        draw.line([p1, p2], fill=color, width=width_line)
        angle = math.atan2(y2 - y1, x2 - x1)
        arr_len = 9
        p_a1 = (x2 - arr_len * math.cos(angle - math.pi / 6), y2 - arr_len * math.sin(angle - math.pi / 6))
        p_a2 = (x2 - arr_len * math.cos(angle + math.pi / 6), y2 - arr_len * math.sin(angle + math.pi / 6))
        draw.polygon([p2, p_a1, p_a2], fill=color)

    # -------------------------------------------------------------
    # LAYOUT GRID (Clean 4-column flow with uncluttered alignment)
    # -------------------------------------------------------------

    # 1. INPUT VIDEO
    draw_box(
        x=50, y=140, w=220, h=110,
        title="Input Video File",
        module_name="Native Stream (≤ 10s)",
        bullets=[
            "• Formats: MP4, AVI, MOV",
            "• Audio-Visual Container"
        ],
        score_str="",
        stroke_color="#2563EB",
        fill_color="#F8FAFC",
        badge="INPUT"
    )

    # Arrow Input -> Stage 1
    draw_arrow((270, 195), (320, 195), "#2563EB")

    # 2. STAGE 1: DEMUXING & PRE-PROCESSING
    draw_box(
        x=320, y=140, w=240, h=110,
        title="Stage 1: Pre-Processing",
        module_name="processor.py",
        bullets=[
            "• 15 FPS Video Sampling",
            "• 16 kHz Mono Audio Demux"
        ],
        score_str="Output: Frames & WAV",
        stroke_color="#0D9488",
        fill_color="#F0FDFA",
        badge="STAGE 1"
    )

    # Split arrows from Stage 1 to Stage 2 (Visual), Stage 3 (Acoustic), Stage 4 (Sync)
    draw_arrow((560, 175), (620, 150), "#4F46E5") # To Visual
    draw_arrow((560, 195), (620, 275), "#7C3AED") # To Acoustic
    draw_arrow((560, 215), (620, 400), "#DB2777") # To Sync

    # 3. STAGE 2: VISUAL BRANCH
    draw_box(
        x=620, y=105, w=300, h=105,
        title="Stage 2: Visual Branch",
        module_name="visual_branch.py",
        bullets=[
            "• MediaPipe Face Crop (+10% margin)",
            "• Vision Transformer (ViT) & Jawline Edge Ratio"
        ],
        score_str="► Visual Score (S_V)",
        stroke_color="#4F46E5",
        fill_color="#EEF2FF",
        badge="NOVELTY 1"
    )

    # 4. STAGE 3: ACOUSTIC BRANCH
    draw_box(
        x=620, y=230, w=300, h=105,
        title="Stage 3: Acoustic Branch",
        module_name="audio_branch.py",
        bullets=[
            "• Wav2Vec2 Synthetic Speech Classifier",
            "• 13 MFCC Variance, Spectral Flatness, ZCR"
        ],
        score_str="► Acoustic Score (S_A)",
        stroke_color="#7C3AED",
        fill_color="#FAF5FF",
        badge="STAGE 3"
    )

    # 5. STAGE 4: LIP-SYNC CONSISTENCY
    draw_box(
        x=620, y=355, w=300, h=105,
        title="Stage 4: Lip-Sync Consistency",
        module_name="consistency_fusion.py",
        bullets=[
            "• 6-Point Inner-Lip MAR vs Acoustic RMS",
            "• VAD-Gated ±400 ms Cross-Correlation"
        ],
        score_str="► Sync Score (S_C)",
        stroke_color="#DB2777",
        fill_color="#FDF2F8",
        badge="NOVELTY 2"
    )

    # Converging arrows from Stages 2, 3, 4 into Stage 5
    draw_arrow((920, 157), (980, 240), "#4F46E5")
    draw_arrow((920, 282), (980, 260), "#7C3AED")
    draw_arrow((920, 407), (980, 280), "#DB2777")

    # 6. STAGE 5: ASYMMETRIC VETO LATE FUSION
    draw_box(
        x=980, y=180, w=330, h=160,
        title="Stage 5: Asymmetric Veto Fusion",
        module_name="consistency_fusion.py",
        bullets=[
            "• Dual-Authentic Gate: Prevents false alarms",
            "• Asymmetric Veto (α = 0.60):",
            "  S_final = 0.60·max(S) + 0.40·Consensus",
            "• Dilution Trap Protection: Single-branch veto"
        ],
        score_str="► Decision: S_final ≥ 0.55 → DEEPFAKE",
        stroke_color="#D97706",
        fill_color="#FFFBEB",
        badge="NOVELTY 3"
    )

    # Arrow Stage 5 -> Stage 6
    draw_arrow((1145, 340), (1145, 500), "#059669")
    draw_arrow((1145, 500), (980, 520), "#059669")

    # 7. STAGE 6: DELIVERY LAYER & INTERACTIVE DASHBOARD
    draw_box(
        x=50, y=490, w=1260, h=140,
        title="Stage 6: Delivery Layer & Interactive Forensic Dashboard",
        module_name="app.py (FastAPI Backend)  &  index.html (Dashboard Interface)",
        bullets=[
            "• Threat Verdict Banner with failing modality attribution (Visual vs Acoustic vs Lip-Sync)",
            "• Real-Time Synchronised Audio-Visual Dynamics Chart (MAR vs RMS) & 6-Point Landmark Mesh Snapshot",
            "• Naive Average (Dilution Trap) vs Context-Aware Asymmetric Veto Side-by-Side Verification",
            "• Explainable 6-Row Signal Telemetry Ledger & One-Click Print-Ready Report Figures Exporter"
        ],
        score_str="► Output: Explainable Forensic Report & Exportable Figures Package (.ZIP)",
        stroke_color="#059669",
        fill_color="#ECFDF5",
        badge="STAGE 6"
    )

    # Save image with 300 DPI
    img.save(output_path, "PNG", dpi=(300, 300))
    print(f"Successfully generated clean minimalist architecture diagram: {output_path}")

if __name__ == "__main__":
    out_file = r"D:\2026\SEM 5\Innovative\Figure_4_1_DEEPSYNC_Stage_Architecture.png"
    create_figure_4_1(out_file)
