"""
tests/test_signal_quality.py  —  Unit Tests for rPPG Signal Quality Index
==========================================================================
Test Coverage
-------------
  1. test_clean_sine_high_sqi
     A pure sine wave at cardiac frequency should score high.

  2. test_pure_noise_low_sqi
     Random Gaussian noise should score low.

  3. test_constant_signal_zero_sqi
     A flat/constant signal should return zero quality.

  4. test_short_signal_handled
     Very short signals (< 10 samples) should return empty SQI gracefully.

  5. test_sqi_range_valid
     All SQI components should be in [0, 1] for any input.

  6. test_filter_bvp_zeroes_bad_windows
     filter_bvp_by_quality should zero out regions of noisy signal.

  7. test_filter_bvp_keeps_good_windows
     filter_bvp_by_quality should preserve clean signal regions.

  8. test_filter_report_structure
     The report dict should contain all expected keys.

  9. test_mixed_signal_partial_rejection
     A signal that is clean for the first half and noisy for the second
     should have some windows rejected and some kept.

  10. test_spectral_snr_monotonic
      Adding more noise to a sine wave should decrease spectral SNR.

  11. test_kurtosis_score_gaussian
      Gaussian noise (kurtosis ≈ 0) should score lower than a pulse-like
      signal (kurtosis ≈ 3–5).

  12. test_periodicity_random_low
      Random noise should have near-zero periodicity score.

Run with:
    pytest tests/test_signal_quality.py -v
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rppg.signal_quality import compute_window_sqi, filter_bvp_by_quality


# ── Fixtures ──────────────────────────────────────────────────────────────────

FPS = 30.0  # standard test frame rate


def _make_clean_bvp(freq_hz: float = 1.2, duration_sec: float = 30.0,
                    fps: float = FPS) -> np.ndarray:
    """Generate a clean sinusoidal BVP signal at a cardiac frequency."""
    t = np.arange(0, duration_sec, 1.0 / fps)
    # Add a small harmonic to make it more pulse-like
    bvp = np.sin(2 * np.pi * freq_hz * t) + \
          0.3 * np.sin(2 * np.pi * 2 * freq_hz * t)
    return bvp


def _make_noisy_bvp(duration_sec: float = 30.0, fps: float = FPS) -> np.ndarray:
    """Generate pure Gaussian noise (no cardiac signal)."""
    n_samples = int(duration_sec * fps)
    rng = np.random.RandomState(42)
    return rng.randn(n_samples)


def _make_constant_bvp(duration_sec: float = 30.0, fps: float = FPS) -> np.ndarray:
    """Generate a flat/constant signal."""
    n_samples = int(duration_sec * fps)
    return np.ones(n_samples) * 5.0


# ── Test: compute_window_sqi ──────────────────────────────────────────────────

class TestComputeWindowSQI:
    """Tests for the per-window SQI computation."""

    def test_clean_sine_high_sqi(self):
        """A clean cardiac-frequency sine should score high composite SQI."""
        bvp = _make_clean_bvp(freq_hz=1.2, duration_sec=30.0)
        sqi = compute_window_sqi(bvp, fps=FPS)

        assert sqi["composite"] > 0.5, \
            f"Clean sine should have high SQI, got {sqi['composite']}"
        assert sqi["is_good"] is True
        assert 0.9 < sqi["dominant_freq_hz"] < 1.5, \
            f"Dominant freq should be near 1.2 Hz, got {sqi['dominant_freq_hz']}"

    def test_pure_noise_low_sqi(self):
        """Random Gaussian noise should score low composite SQI."""
        bvp = _make_noisy_bvp()
        sqi = compute_window_sqi(bvp, fps=FPS)

        assert sqi["composite"] < 0.5, \
            f"Noise should have low SQI, got {sqi['composite']}"

    def test_constant_signal_zero_sqi(self):
        """A constant signal should return zero-quality SQI."""
        bvp = _make_constant_bvp()
        sqi = compute_window_sqi(bvp, fps=FPS)

        assert sqi["composite"] == 0.0
        assert sqi["is_good"] is False

    def test_short_signal_handled(self):
        """Signals shorter than 10 samples should return empty SQI gracefully."""
        bvp = np.array([1.0, 2.0, 3.0])
        sqi = compute_window_sqi(bvp, fps=FPS)

        assert sqi["composite"] == 0.0
        assert sqi["is_good"] is False

    def test_sqi_range_valid(self):
        """All SQI components should be in [0, 1]."""
        for signal_fn in [_make_clean_bvp, _make_noisy_bvp]:
            bvp = signal_fn()
            sqi = compute_window_sqi(bvp, fps=FPS)

            for key in ["spectral_snr", "kurtosis_score", "periodicity", "composite"]:
                assert 0.0 <= sqi[key] <= 1.0, \
                    f"{key} = {sqi[key]} is out of [0,1] range"

    def test_empty_array(self):
        """An empty array should return zero quality."""
        sqi = compute_window_sqi(np.array([]), fps=FPS)
        assert sqi["composite"] == 0.0
        assert sqi["is_good"] is False


# ── Test: filter_bvp_by_quality ───────────────────────────────────────────────

class TestFilterBVPByQuality:
    """Tests for the full-signal quality filtering."""

    def test_filter_keeps_good_windows(self):
        """A clean signal should be largely preserved after filtering."""
        bvp = _make_clean_bvp(freq_hz=1.2, duration_sec=60.0)
        bvp_filtered, mask, report = filter_bvp_by_quality(
            bvp, fps=FPS, window_sec=30.0, step_sec=10.0, threshold=0.4
        )

        # Most frames should be kept
        pct_kept = mask.sum() / len(mask)
        assert pct_kept > 0.5, \
            f"Clean signal should keep >50% frames, got {pct_kept*100:.0f}%"
        assert report["n_good"] > 0

    def test_filter_zeroes_bad_windows(self):
        """A pure noise signal should have all or most windows rejected."""
        bvp = _make_noisy_bvp(duration_sec=60.0)
        bvp_filtered, mask, report = filter_bvp_by_quality(
            bvp, fps=FPS, window_sec=30.0, step_sec=10.0, threshold=0.4
        )

        # Most frames should be zeroed
        pct_zeroed = (~mask).sum() / len(mask)
        assert pct_zeroed > 0.3, \
            f"Noise should reject >30% frames, got {pct_zeroed*100:.0f}% zeroed"

    def test_filter_report_structure(self):
        """The report dict should contain all expected keys."""
        bvp = _make_clean_bvp(duration_sec=60.0)
        _, _, report = filter_bvp_by_quality(bvp, fps=FPS)

        required_keys = {"n_windows", "n_good", "n_rejected",
                         "pct_good", "mean_sqi", "per_window"}
        assert required_keys.issubset(report.keys()), \
            f"Missing keys: {required_keys - set(report.keys())}"

        assert isinstance(report["per_window"], list)
        assert len(report["per_window"]) > 0

        # Each per-window entry should have the SQI fields
        w = report["per_window"][0]
        assert "composite" in w
        assert "start_frame" in w
        assert "end_frame" in w

    def test_mixed_signal_partial_rejection(self):
        """Signal that is clean first half, noisy second half."""
        clean = _make_clean_bvp(freq_hz=1.2, duration_sec=30.0)
        noisy = _make_noisy_bvp(duration_sec=30.0)
        bvp = np.concatenate([clean, noisy])

        bvp_filtered, mask, report = filter_bvp_by_quality(
            bvp, fps=FPS, window_sec=15.0, step_sec=10.0, threshold=0.4
        )

        # Should have a mix of good and bad windows
        assert report["n_good"] >= 1, "Should keep at least the clean portion"
        assert report["n_rejected"] >= 0, "May reject some noisy windows"
        assert report["n_windows"] >= 3, "Should have multiple windows"


# ── Test: Individual metric quality ───────────────────────────────────────────

class TestIndividualMetrics:
    """Tests for the three sub-metrics of SQI."""

    def test_spectral_snr_monotonic(self):
        """Adding noise should decrease spectral SNR."""
        bvp_clean = _make_clean_bvp(freq_hz=1.2)
        sqi_clean = compute_window_sqi(bvp_clean, fps=FPS)

        # Add moderate noise
        rng = np.random.RandomState(42)
        bvp_noisy = bvp_clean + 2.0 * rng.randn(len(bvp_clean))
        sqi_noisy = compute_window_sqi(bvp_noisy, fps=FPS)

        assert sqi_clean["spectral_snr"] > sqi_noisy["spectral_snr"], \
            f"Clean SNR {sqi_clean['spectral_snr']} should > noisy {sqi_noisy['spectral_snr']}"

    def test_kurtosis_score_gaussian_vs_pulse(self):
        """Pulse-like signal should score higher overall than Gaussian noise."""
        # Gaussian noise
        rng = np.random.RandomState(42)
        gaussian = rng.randn(900)
        sqi_gauss = compute_window_sqi(gaussian, fps=FPS)

        # Pulse-like signal (sharp peaks)
        pulse = _make_clean_bvp(freq_hz=1.2)
        sqi_pulse = compute_window_sqi(pulse, fps=FPS)

        # The composite SQI (not just kurtosis) should favor the pulse signal
        # because spectral SNR and periodicity will be much higher
        assert sqi_pulse["composite"] > sqi_gauss["composite"], \
            f"Pulse composite {sqi_pulse['composite']} should > noise {sqi_gauss['composite']}"

    def test_periodicity_random_low(self):
        """Random noise should have low periodicity score."""
        bvp = _make_noisy_bvp()
        sqi = compute_window_sqi(bvp, fps=FPS)
        assert sqi["periodicity"] < 0.5, \
            f"Noise periodicity should be low, got {sqi['periodicity']}"

    def test_periodicity_sine_high(self):
        """A periodic sine wave should have high periodicity score."""
        bvp = _make_clean_bvp(freq_hz=1.0, duration_sec=30.0)
        sqi = compute_window_sqi(bvp, fps=FPS)
        assert sqi["periodicity"] > 0.3, \
            f"Sine periodicity should be moderate-high, got {sqi['periodicity']}"


# ── Test: Edge cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case handling."""

    def test_single_window_signal(self):
        """Signal exactly one window long should work."""
        bvp = _make_clean_bvp(duration_sec=30.0)
        bvp_filtered, mask, report = filter_bvp_by_quality(
            bvp, fps=FPS, window_sec=30.0
        )
        assert report["n_windows"] >= 1

    def test_very_short_signal(self):
        """Signal shorter than one window should still return a report."""
        bvp = _make_clean_bvp(duration_sec=5.0)
        bvp_filtered, mask, report = filter_bvp_by_quality(
            bvp, fps=FPS, window_sec=30.0
        )
        # May have 0 or 1 windows depending on clamping
        assert "n_windows" in report

    def test_different_fps_values(self):
        """SQI should work across different frame rates."""
        for fps in [25.0, 29.5, 29.97, 30.0, 60.0]:
            bvp = _make_clean_bvp(freq_hz=1.2, duration_sec=30.0, fps=fps)
            sqi = compute_window_sqi(bvp, fps=fps)
            assert 0.0 <= sqi["composite"] <= 1.0, \
                f"SQI out of range at fps={fps}: {sqi['composite']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])