"""
report_generator.py  -  MOXIE-VASA PDF Report Generator
=========================================================
Generates a 3-page PDF stress report for one subject using fpdf2.

Page 1 - Subject overview + stress score gauge
Page 2 - Heart rate timeline + AU feature bar chart
Page 3 - Indicator breakdown table + methodology note

Usage
-----
    from report_generator import generate_report

    generate_report(
        score_result  = stress_inference.compute_stress_score(features),
        features      = features,
        bvp_csv_path  = "Pipeline_Output/result_rppg/subject1_vid_safe_bvp.csv",
        openface_csv  = "Pipeline_Output/result_openface/subject1_vid_safe.csv",
        output_path   = "Pipeline_Output/reports/subject1_report.pdf",
        fps           = 29.26,
    )
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF


# -- Colour palette ------------------------------------------------------------
BLUE       = (44,  95,  138)
RED        = (192, 57,  43)
GREEN      = (39,  174, 96)
ORANGE     = (230, 126, 34)
LIGHT_GREY = (245, 245, 245)
DARK_GREY  = (80,  80,  80)
WHITE      = (255, 255, 255)

def _score_colour(score: float) -> tuple:
    if score < 30:
        return GREEN
    elif score < 60:
        return ORANGE
    else:
        return RED

def _classification_colour(classification: str) -> tuple:
    return {"LOW STRESS": GREEN,
            "MODERATE STRESS": ORANGE,
            "HIGH STRESS": RED}.get(classification, DARK_GREY)


# -- Plot helpers --------------------------------------------------------------

def _make_hr_plot(bvp_csv: Path, fps: float) -> Optional[bytes]:
    """Generate HR timeline plot, return as PNG bytes."""
    try:
        from scipy.signal import find_peaks
        bvp  = pd.read_csv(bvp_csv)["bvp"].values
        time = np.arange(len(bvp)) / fps

        peaks, _ = find_peaks(bvp, distance=int(fps * 0.4),
                               height=bvp.mean() + bvp.std() * 0.3)

        fig, axes = plt.subplots(2, 1, figsize=(9, 4.5))
        fig.patch.set_facecolor("white")

        # BVP waveform
        axes[0].plot(time, bvp, color="#2C5F8A", linewidth=0.7, alpha=0.9)
        axes[0].scatter(time[peaks], bvp[peaks], s=12,
                        color="#C0392B", zorder=5)
        axes[0].set_ylabel("BVP amplitude", fontsize=9)
        axes[0].set_title("Blood Volume Pulse (rPPG)", fontsize=10,
                           fontweight="bold")
        axes[0].set_xlim(0, time[-1])
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # HR over time
        if len(peaks) >= 4:
            rr_ms   = (np.diff(peaks) / fps) * 1000
            hr_inst = 60000 / rr_ms
            valid   = (hr_inst > 40) & (hr_inst < 200)
            hr_v    = hr_inst[valid]
            t_v     = time[peaks[1:]][valid]

            axes[1].plot(t_v, hr_v, color="#2C5F8A", linewidth=1.2,
                         marker="o", markersize=3,
                         markerfacecolor="white",
                         markeredgecolor="#2C5F8A",
                         label="Instantaneous HR")
            axes[1].axhline(hr_v.mean(), color="#C0392B",
                            linewidth=1, linestyle="--",
                            label=f"Mean = {hr_v.mean():.1f} bpm")
            axes[1].axhspan(60, 100, alpha=0.06, color="#27AE60",
                            label="Normal range")
            axes[1].set_ylim(40, 160)
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "Insufficient signal for HR timeline",
                         ha="center", va="center",
                         transform=axes[1].transAxes, fontsize=9)

        axes[1].set_xlabel("Time (seconds)", fontsize=9)
        axes[1].set_ylabel("Heart Rate (bpm)", fontsize=9)
        axes[1].set_title("Heart Rate Over Time", fontsize=10,
                           fontweight="bold")
        axes[1].set_xlim(0, time[-1])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"[Report] HR plot error: {e}")
        return None


def _make_au_plot(features: dict) -> Optional[bytes]:
    """Generate AU bar chart, return as PNG bytes."""
    try:
        au_items = [
            ("AU04\nBrow furrow",    features.get("au04_mean", 0)),
            ("AU07\nLid tighten",   features.get("au07_mean", 0)),
            ("AU15\nLip depress",   features.get("au15_mean", 0)),
            ("AU23\nLip press",     features.get("au23_mean", 0)),
            ("AU05\nUpper lid",     features.get("au05_mean", 0)),
            ("AU12\nSmile",         features.get("au12_mean", 0)),
        ]
        labels, values = zip(*au_items)

        # Colour: red if above threshold, blue otherwise
        thresholds = [1.0, 0.5, 0.8, 0.8, 0.0, 0.0]
        colors = ["#C0392B" if v > t else "#2C5F8A"
                  for v, t in zip(values, thresholds)]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("white")
        bars = ax.bar(labels, values, color=colors,
                      edgecolor="white", alpha=0.9)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", fontsize=9)

        ax.set_ylabel("Mean Intensity (0-5 scale)", fontsize=10)
        ax.set_title("Facial Action Unit Intensities",
                     fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(max(values) * 1.3, 2.5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        red_patch  = mpatches.Patch(color="#C0392B", label="Above threshold")
        blue_patch = mpatches.Patch(color="#2C5F8A", label="Below threshold")
        ax.legend(handles=[red_patch, blue_patch], fontsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"[Report] AU plot error: {e}")
        return None


# -- PDF class -----------------------------------------------------------------

class StressReportPDF(FPDF):

    def header(self):
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 210, 14, "F")
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*WHITE)
        self.set_xy(8, 3)
        self.cell(0, 8, "MOXIE-VASA  |  Video-Based Stress Assessment Report")
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*DARK_GREY)
        self.cell(0, 6,
                  "MOXIE-VASA Phase 1  |  Rule-based scoring  |  "
                  "For research use only  |  "
                  f"Page {self.page_no()}",
                  align="C")
        self.set_text_color(0, 0, 0)

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(*LIGHT_GREY)
        self.set_text_color(*BLUE)
        self.cell(0, 8, f"  {text}", ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kv_row(self, label: str, value: str, highlight: bool = False):
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(*LIGHT_GREY)
        self.cell(70, 7, f"  {label}", fill=True, border=0)
        self.set_font("Helvetica", "", 10)
        fill_col = (255, 240, 240) if highlight else WHITE
        self.set_fill_color(*fill_col)
        self.cell(0, 7, f"  {value}", fill=True, border=0, ln=True)
        self.ln(1)


# -- Main function -------------------------------------------------------------

def generate_report(
    score_result:  dict,
    features:      dict,
    output_path:   str | Path,
    bvp_csv_path:  Optional[str | Path] = None,
    fps:           float = 30.0,
) -> bool:
    """
    Generate a PDF stress report for one subject.

    Parameters
    ----------
    score_result  : dict from stress_inference.compute_stress_score()
    features      : dict from feature_extractor.extract_features()
    output_path   : path to write the PDF
    bvp_csv_path  : path to BVP CSV (for HR plot); optional
    fps           : video frame rate

    Returns
    -------
    bool - True if PDF was written successfully
    """
    try:
        from fpdf import FPDF
    except ImportError:
        print("[Report] fpdf2 not installed. Run: pip install fpdf2")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subject_id     = score_result.get("subject_id", "unknown")
    score          = score_result.get("score", 0)
    classification = score_result.get("classification", "UNKNOWN")
    confidence     = score_result.get("confidence", "LOW")
    indicators     = score_result.get("indicators", [])
    hrv_note       = score_result.get("hrv_note", {})
    warnings       = score_result.get("warnings", [])
    score_color    = _score_colour(score)

    pdf = StressReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 18, 10)

    # -- PAGE 1: Overview and score --------------------------------------------
    pdf.add_page()
    pdf.ln(4)

    # Subject info
    pdf.section_title("Subject Information")
    duration = features.get("duration_s", 0)
    n_frames = features.get("n_openface_frames", 0)
    pdf.kv_row("Subject ID",       subject_id)
    pdf.kv_row("Video Duration",   f"{duration:.1f} seconds")
    pdf.kv_row("OpenFace Frames",  str(n_frames))
    pdf.kv_row("Signal Confidence", confidence,
               highlight=(confidence == "LOW"))
    pdf.ln(4)

    # Stress score - large display
    pdf.section_title("Stress Assessment")
    pdf.ln(2)

    # Score box
    pdf.set_fill_color(*score_color)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 36)
    pdf.cell(0, 22, f"{score:.0f}%", align="C", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*_classification_colour(classification))
    pdf.cell(0, 10, classification, align="C", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Key physiological values
    pdf.section_title("Key Physiological Indicators")
    hr     = features.get("hr_mean", 0)
    rmssd  = features.get("rmssd", 0)
    lf_hf  = features.get("lf_hf_ratio", 0)
    gt_hr  = features.get("gt_hr")
    hr_mae = features.get("hr_mae")

    pdf.kv_row("Mean Heart Rate (rPPG)",
               f"{hr:.1f} bpm  {'[!] Elevated' if hr > 85 else '[OK] Normal range'}",
               highlight=(hr > 85))
    if gt_hr:
        pdf.kv_row("Ground Truth HR (oximeter)",
                   f"{gt_hr:.1f} bpm  |  MAE = {hr_mae:.1f} bpm")
    pdf.kv_row("RMSSD (HRV)",
               f"{rmssd:.1f} ms  "
               f"[Note: Phase 1 values unreliable - tuning pending]")
    pdf.kv_row("LF/HF Ratio",
               f"{lf_hf:.3f}  "
               f"{'[!] Elevated (sympathetic)' if lf_hf > 2.0 else 'Within range'}")
    pdf.ln(4)

    if warnings:
        pdf.section_title("Quality Warnings")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*RED)
        for w in warnings:
            pdf.multi_cell(0, 5, f"  [!] {w}")
        pdf.set_text_color(0, 0, 0)

    # -- PAGE 2: HR plot + AU chart --------------------------------------------
    pdf.add_page()
    pdf.ln(4)

    # HR timeline plot
    if bvp_csv_path and Path(bvp_csv_path).exists():
        hr_png = _make_hr_plot(Path(bvp_csv_path), fps)
        if hr_png:
            tmp = Path("/tmp/moxie_hr_plot.png")
            tmp.write_bytes(hr_png)
            pdf.section_title("Heart Rate Analysis (rPPG)")
            pdf.image(str(tmp), x=10, w=185)
            pdf.ln(4)

    # AU bar chart
    au_png = _make_au_plot(features)
    if au_png:
        tmp_au = Path("/tmp/moxie_au_plot.png")
        tmp_au.write_bytes(au_png)
        pdf.section_title("Facial Action Unit Analysis (OpenFace)")
        pdf.image(str(tmp_au), x=10, w=185)
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*DARK_GREY)
        pdf.multi_cell(0, 4,
            "AU04=Brow furrow (stress/concentration)  "
            "AU07=Lid tighten (tension)  "
            "AU15=Lip depress (negative affect)  "
            "AU23=Lip press (cognitive effort)  "
            "AU12=Smile (inverse stress indicator)")
        pdf.set_text_color(0, 0, 0)

    # -- PAGE 3: Indicators + methodology -------------------------------------
    pdf.add_page()
    pdf.ln(4)

    # Indicator table
    pdf.section_title("Stress Indicator Breakdown")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*BLUE)
    pdf.set_text_color(*WHITE)
    pdf.cell(65, 7, "  Indicator",       fill=True)
    pdf.cell(22, 7, "Value",  align="C", fill=True)
    pdf.cell(22, 7, "Thresh", align="C", fill=True)
    pdf.cell(18, 7, "Weight", align="C", fill=True)
    pdf.cell(22, 7, "Contrib", align="C", fill=True)
    pdf.cell(36, 7, "Status", align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for i, ind in enumerate(indicators):
        fill = LIGHT_GREY if i % 2 == 0 else WHITE
        pdf.set_fill_color(*fill)
        triggered = ind["triggered"]
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(65, 6, f"  {ind['label'][:38]}",      fill=True)
        pdf.cell(22, 6, f"{ind['value']:.2f}", align="C", fill=True)
        pdf.cell(22, 6, f"{ind['threshold']:.1f}", align="C", fill=True)
        pdf.cell(18, 6, f"{int(ind['weight']*100)}%", align="C", fill=True)
        pdf.cell(22, 6, f"{ind['contribution']:.1f}%", align="C", fill=True)
        status = "TRIGGERED" if triggered else "not triggered"
        pdf.set_text_color(*RED if triggered else DARK_GREY)
        pdf.cell(36, 6, status, align="C", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    pdf.ln(4)

    # HRV note
    pdf.section_title("HRV Status (Phase 1 Note)")
    pdf.set_font("Helvetica", "", 9)
    note = hrv_note.get("reliability_note", "")
    pdf.multi_cell(0, 5, f"  {note}")
    pdf.ln(2)

    # Methodology
    pdf.section_title("Methodology & Limitations")
    pdf.set_font("Helvetica", "", 9)
    methodology = (
        "Heart rate is estimated using remote photoplethysmography (rPPG) "
        "with the POS algorithm (Wang et al., 2017). Facial Action Units are "
        "extracted by OpenFace (Baltrusaitis et al., 2018) using the 68-point "
        "iBUG landmark model. The stress score is a rule-based weighted sum "
        "of physiological and facial indicators grounded in published stress "
        "physiology literature. This is NOT a validated clinical tool. "
        "The score reflects relative stress indicators from video only and "
        "should be interpreted alongside other clinical measures. "
        "HRV (RMSSD, LF/HF) is computed but excluded from the score in "
        "Phase 1 pending peak detector tuning. "
        "Phase 3 will replace this rule-based scorer with a trained "
        "Random Forest classifier using UBFC-Phys labeled data."
    )
    pdf.multi_cell(0, 5, methodology)
    pdf.ln(3)

    pdf.section_title("References")
    pdf.set_font("Helvetica", "", 8)
    refs = [
        "Wang, W. et al. (2017). Algorithmic Principles of Remote PPG. "
        "IEEE Trans. Biomed. Eng., 64(7), 1479-1491.",
        "Baltrusaitis, T. et al. (2018). OpenFace 2.0. IEEE FG.",
        "Taelman, J. et al. (2009). Influence of mental stress on HR and HRV. "
        "IFMBE Proceedings, 22.",
        "Ekman, P. & Friesen, W. (1978). Facial Action Coding System. "
        "Consulting Psychologists Press.",
    ]
    for ref in refs:
        pdf.multi_cell(0, 4, f"  * {ref}")
        pdf.ln(1)

    pdf.output(str(output_path))
    print(f"[Report] PDF saved -> {output_path}")
    return True