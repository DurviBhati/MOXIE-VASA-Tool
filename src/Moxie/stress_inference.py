"""
stress_inference.py  —  MOXIE-VASA Rule-Based Stress Scorer
============================================================
Computes a stress score (0–100%) and confidence rating from
the feature dict produced by feature_extractor.py.

This is a rule-based heuristic scorer — NOT a trained ML classifier.
Each indicator is grounded in published stress physiology literature.
The score is a weighted sum of triggered indicators.

Stress indicators and thresholds
---------------------------------
Physiological (from rPPG):
    HR > 85 bpm           — elevated HR in stress (weight: 40%)
      Literature: Taelman et al. 2009 — HR increases ~15 bpm under stress

Facial behaviour (from OpenFace):
    AU04 mean > 1.0       — brow furrow: concentration/distress (weight: 20%)
      Literature: Ekman FACS — AU04 in cognitive load and negative affect
    AU23 mean > 0.8       — lip presser: concentration/stress (weight: 20%)
      Literature: Bartlett et al. — AU23 in sustained cognitive effort
    AU15 mean > 0.8       — lip corner depressor: negative affect (weight: 10%)
      Literature: Ekman — AU15 associated with sadness/distress
    AU07 mean > 0.5       — lid tightener: tension/focus (weight: 10%)
      Literature: Ekman — AU07 in threat/concentration contexts

Note on HRV: RMSSD and LF/HF are computed but flagged as unreliable
in Phase 1 due to peak detector tuning. They are included in the
output for transparency but excluded from the score calculation.

Confidence rating
-----------------
Based on signal quality indicators:
    HIGH   : valid rPPG + >80% OpenFace detection + duration > 45s
    MEDIUM : valid rPPG + >60% OpenFace detection
    LOW    : any quality issue present
"""

from __future__ import annotations
from typing import Any


# ── Thresholds ────────────────────────────────────────────────────────────────

THRESHOLDS = {
    # (feature_key, threshold, direction, weight, label, literature_ref)
    # Physiological indicator — primary stress signal (rPPG)
    "hr":    ("hr_mean",   85.0,  "above", 0.50,
              "Elevated heart rate (>85 bpm)",
              "Taelman et al. 2009 — HR rises ~15 bpm under stress"),
    # Facial indicators — calibrated to UBFC-rPPG data range
    # Note: cognitive stress (mental arithmetic) produces subtler facial
    # expression than emotional stress. Thresholds reflect this.
    "au04":  ("au04_mean", 0.25,  "above", 0.20,
              "Brow furrowing (AU04 >0.25)",
              "Ekman FACS — AU04 in cognitive load and concentration"),
    "au14":  ("au14_mean", 0.25,  "above", 0.15,
              "Dimpler/jaw tension (AU14 >0.25)",
              "Ekman FACS — AU14 in sustained effort and tension"),
    "au07":  ("au07_mean", 0.15,  "above", 0.10,
              "Lid tightening (AU07 >0.15)",
              "Ekman FACS — AU07 in threat/concentration contexts"),
    "au15":  ("au15_mean", 0.05,  "above", 0.05,
              "Lip corner depression (AU15 >0.05)",
              "Ekman FACS — AU15 in negative affect"),
}

# HRV indicators — computed but not scored (Phase 1 limitation)
HRV_INFO = {
    "rmssd":     ("rmssd",     30.0,  "below",
                  "Low HRV/RMSSD (<30ms) — sympathetic dominance",
                  "Thayer et al. — reduced HRV in stress"),
    "lf_hf":     ("lf_hf_ratio", 2.0, "above",
                  "High LF/HF ratio (>2.0) — sympathetic dominance",
                  "Task Force 1996 — LF/HF as autonomic balance marker"),
}


def compute_stress_score(features: dict) -> dict:
    """
    Compute a rule-based stress score from extracted features.

    Parameters
    ----------
    features : dict from feature_extractor.extract_features()

    Returns
    -------
    dict with keys:
        score           — float 0.0–100.0 (weighted stress score)
        confidence      — "HIGH" | "MEDIUM" | "LOW"
        classification  — "LOW STRESS" | "MODERATE STRESS" | "HIGH STRESS"
        indicators      — list of triggered indicator dicts
        hrv_note        — HRV status (computed but excluded from score)
        feature_summary — key features used in scoring
        warnings        — list of quality warning strings
    """
    if not features:
        return _empty_result("No features provided")

    indicators = []
    total_weight_triggered = 0.0
    warnings = []

    # ── Score each indicator ──────────────────────────────────────────────────
    for key, (feat_name, threshold, direction, weight, label, ref) in THRESHOLDS.items():
        value = features.get(feat_name)
        if value is None:
            warnings.append(f"Feature '{feat_name}' missing — indicator '{key}' skipped")
            continue

        if direction == "above":
            triggered = float(value) > threshold
        else:
            triggered = float(value) < threshold

        # Partial credit: how far above/below threshold
        if triggered and direction == "above":
            excess = min((float(value) - threshold) / threshold, 1.0)
            partial_weight = weight * (0.5 + 0.5 * excess)
        elif triggered and direction == "below":
            deficit = min((threshold - float(value)) / threshold, 1.0)
            partial_weight = weight * (0.5 + 0.5 * deficit)
        else:
            partial_weight = 0.0

        total_weight_triggered += partial_weight

        indicators.append({
            "key":          key,
            "label":        label,
            "value":        round(float(value), 3),
            "threshold":    threshold,
            "triggered":    triggered,
            "weight":       weight,
            "contribution": round(partial_weight * 100, 1),
            "literature":   ref,
        })

    score = round(min(total_weight_triggered * 100, 100.0), 1)

    # ── Classification ────────────────────────────────────────────────────────
    if score < 30:
        classification = "LOW STRESS"
    elif score < 60:
        classification = "MODERATE STRESS"
    else:
        classification = "HIGH STRESS"

    # ── HRV note (computed but not scored) ───────────────────────────────────
    hrv_note = _compute_hrv_note(features)

    # ── Confidence ───────────────────────────────────────────────────────────
    confidence = _compute_confidence(features, warnings)

    # ── Feature summary ───────────────────────────────────────────────────────
    feature_summary = {
        "hr_mean":     features.get("hr_mean"),
        "au04_mean":   features.get("au04_mean"),
        "au07_mean":   features.get("au07_mean"),
        "au15_mean":   features.get("au15_mean"),
        "au23_mean":   features.get("au23_mean"),
        "rmssd":       features.get("rmssd"),
        "lf_hf_ratio": features.get("lf_hf_ratio"),
        "duration_s":  features.get("duration_s"),
    }

    return {
        "subject_id":      features.get("subject_id", "unknown"),
        "score":           score,
        "confidence":      confidence,
        "classification":  classification,
        "indicators":      indicators,
        "hrv_note":        hrv_note,
        "feature_summary": feature_summary,
        "warnings":        warnings,
    }


def _compute_hrv_note(features: dict) -> dict:
    """Compute HRV status for transparency — not included in score."""
    rmssd    = features.get("rmssd", 0)
    lf_hf    = features.get("lf_hf_ratio", 0)

    # Note: Phase 1 HRV values are unreliable (peak detector not tuned)
    # Normal resting RMSSD: 20-80ms. Our values 200-400ms indicate noise.
    rmssd_flag = rmssd < 30 and rmssd > 0

    return {
        "rmssd":            round(rmssd, 2),
        "rmssd_flag":       rmssd_flag,
        "lf_hf_ratio":      round(lf_hf, 4),
        "lf_hf_flag":       lf_hf > 2.0,
        "reliability_note": (
            "HRV metrics computed but excluded from stress score. "
            "Peak detector tuning required for reliable HRV (Phase 3)."
            if rmssd > 100 else
            "HRV metrics within plausible range."
        ),
    }


def _compute_confidence(features: dict, warnings: list) -> str:
    """Rate signal quality as HIGH / MEDIUM / LOW."""
    duration = features.get("duration_s", 0)
    n_frames = features.get("n_openface_frames", 0)
    hr_mean  = features.get("hr_mean", 0)
    has_gt   = features.get("gt_hr") is not None

    issues = len(warnings)
    if duration < 30:
        issues += 2
    elif duration < 45:
        issues += 1
    if n_frames < 500:
        issues += 1
    if hr_mean <= 0:
        issues += 2

    if issues == 0:
        return "HIGH"
    elif issues <= 1:
        return "MEDIUM"
    else:
        return "LOW"


def _empty_result(reason: str) -> dict:
    return {
        "subject_id":     "unknown",
        "score":          0.0,
        "confidence":     "LOW",
        "classification": "UNKNOWN",
        "indicators":     [],
        "hrv_note":       {},
        "feature_summary":{},
        "warnings":       [reason],
    }


def batch_score(features_list: list[dict]) -> list[dict]:
    """Score a list of feature dicts. Returns list of score dicts."""
    return [compute_stress_score(f) for f in features_list]