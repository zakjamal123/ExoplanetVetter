"""Tests for tools/diagnostics.py.

Two targets:
  - TrES-2b (K00001.01 / KIC 11446443): confirmed hot Jupiter — all flags must be False.
  - Synthetic half-period EB: primary (10 % depth) + secondary (5 % depth) at phase 0.5 —
    must trip odd/even, secondary-eclipse, and radius checks.
"""
import numpy as np
import pytest
from lightkurve import LightCurve

from tools.diagnostics import (
    OddEvenResult,
    RadiusSanityResult,
    SecondaryEclipseResult,
    ShapeMetricResult,
    odd_even_depth_test,
    radius_sanity,
    secondary_eclipse_test,
    shape_metric,
)

# ── TrES-2b catalog constants ────────────────────────────────────────────────
CATALOG_PERIOD = 2.470613        # days
CATALOG_T0 = 122.763305          # BKJD
CATALOG_DURATION = 1.74319 / 24  # hours → days
CATALOG_PRAD = 13.04             # R_Earth  (koi_prad; 1.16 R_Jup → no radius flag)

# ── Synthetic EB parameters ──────────────────────────────────────────────────
# True period = 6 days.  BLS would find 3 days (half), so the secondary
# eclipse appears at phase 0.5 and every other "transit" is the secondary.
EB_TRUE_PERIOD = 6.0
EB_PERIOD = EB_TRUE_PERIOD / 2      # reported half-period
EB_T0 = 1.0
EB_DURATION = 0.15                  # days
EB_PRIMARY_DEPTH = 0.10             # 10 % → rp/rs ≈ 0.316
EB_SECONDARY_DEPTH = 0.05           # 5 %  → odd/even mismatch
_R_SUN_IN_REARTH = 109.076
# With Rs = 1 Rsun, Rp = sqrt(0.10) * 109 ≈ 34.5 R_Earth ≈ 3.1 R_Jup → radius flag
EB_PLANET_RADIUS_REARTH = float(np.sqrt(EB_PRIMARY_DEPTH) * _R_SUN_IN_REARTH)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_synthetic_eb() -> LightCurve:
    """Box-transit EB: primary at phase 0, secondary at phase 0.5 of EB_PERIOD."""
    rng = np.random.default_rng(42)
    n = 10_000
    t = np.linspace(0.0, 100.0, n)
    f = np.ones(n)

    h = EB_DURATION / 2.0
    # Phase relative to true period
    phase_true = (t - EB_T0) % EB_TRUE_PERIOD

    # Primary eclipses: at phase_true ≈ 0
    f[(phase_true < h) | (phase_true > EB_TRUE_PERIOD - h)] -= EB_PRIMARY_DEPTH
    # Secondary eclipses: at phase_true ≈ EB_TRUE_PERIOD / 2
    f[np.abs(phase_true - EB_TRUE_PERIOD / 2.0) < h] -= EB_SECONDARY_DEPTH

    f += rng.normal(0.0, 1e-3, n)
    return LightCurve(time=t, flux=f)


@pytest.fixture(scope="module")
def eb_lc() -> LightCurve:
    return _make_synthetic_eb()


# ── Confirmed planet: TrES-2b ─────────────────────────────────────────────────

def test_odd_even_confirmed_planet(flat_lc):
    result = odd_even_depth_test(flat_lc, CATALOG_PERIOD, CATALOG_T0, CATALOG_DURATION)
    assert not result["flag"], (
        f"Unexpected odd/even flag on confirmed planet TrES-2b: {result}"
    )
    assert result["depth_ratio"] < 0.10, (
        f"Odd/even ratio {result['depth_ratio']:.4f} too large for confirmed planet"
    )


def test_secondary_eclipse_confirmed_planet(flat_lc):
    result = secondary_eclipse_test(flat_lc, CATALOG_PERIOD, CATALOG_T0, CATALOG_DURATION)
    assert not result["flag"], (
        f"Unexpected secondary eclipse flag on TrES-2b: {result}"
    )
    assert result["significance"] < 3.0, (
        f"Secondary significance {result['significance']:.2f}σ too high for confirmed planet"
    )


def test_shape_metric_confirmed_planet(flat_lc):
    result = shape_metric(flat_lc, CATALOG_PERIOD, CATALOG_T0, CATALOG_DURATION)
    assert not result["flag"], (
        f"Unexpected V-shape flag on TrES-2b: {result}"
    )
    assert result["vshape_ratio"] < 0.4, (
        f"vshape_ratio {result['vshape_ratio']:.3f} too high for a U-shape transit"
    )


def test_radius_sanity_confirmed_planet():
    result = radius_sanity(CATALOG_PRAD)
    assert not result["flag"], (
        f"Radius flag on confirmed planet (koi_prad={CATALOG_PRAD} R_Earth): {result}"
    )
    assert result["planet_radius_rjup"] < 2.0


# ── Synthetic eclipsing binary ────────────────────────────────────────────────

def test_eb_odd_even_flag(eb_lc):
    result = odd_even_depth_test(eb_lc, EB_PERIOD, EB_T0, EB_DURATION)
    assert result["flag"], (
        f"Odd/even test failed to flag synthetic EB: {result}\n"
        f"Expected odd≈{EB_PRIMARY_DEPTH}, even≈{EB_SECONDARY_DEPTH}."
    )
    # Verify the numeric values are in the right ballpark
    assert abs(result["odd_depth"] - EB_PRIMARY_DEPTH) < 0.02
    assert abs(result["even_depth"] - EB_SECONDARY_DEPTH) < 0.02


def test_eb_secondary_eclipse_flag(eb_lc):
    # The secondary eclipse is at t0 + TRUE_PERIOD/2.  Folding at the TRUE period
    # places it at phase 0.5.  (Folding at the BLS half-period folds both
    # eclipses onto phase 0, so odd_even_depth_test is the correct check there.)
    result = secondary_eclipse_test(eb_lc, EB_TRUE_PERIOD, EB_T0, EB_DURATION)
    assert result["flag"], (
        f"Secondary eclipse test failed to flag synthetic EB at true period "
        f"({EB_TRUE_PERIOD} d): {result}"
    )
    assert result["secondary_depth"] > 0.03, (
        f"Secondary depth {result['secondary_depth']:.4f} unexpectedly shallow"
    )
    assert result["significance"] > 10.0, (
        f"Secondary significance {result['significance']:.1f}σ unexpectedly low"
    )


def test_eb_radius_flag():
    result = radius_sanity(EB_PLANET_RADIUS_REARTH)
    assert result["flag"], (
        f"Radius sanity failed to flag synthetic EB "
        f"(Rp={EB_PLANET_RADIUS_REARTH:.1f} R_Earth = {result['planet_radius_rjup']:.2f} R_Jup): {result}"
    )
    assert result["planet_radius_rjup"] > 2.0


def test_eb_trips_at_least_one_flag(eb_lc):
    """Regression guard: if this fails, diagnostics carry no discriminative signal."""
    oe = odd_even_depth_test(eb_lc, EB_PERIOD, EB_T0, EB_DURATION)
    sec = secondary_eclipse_test(eb_lc, EB_PERIOD, EB_T0, EB_DURATION)
    rad = radius_sanity(EB_PLANET_RADIUS_REARTH)

    any_flag = oe["flag"] or sec["flag"] or rad["flag"]
    assert any_flag, (
        "Synthetic EB tripped no diagnostic — diagnostics are not discriminative.\n"
        f"  odd/even:  {oe}\n"
        f"  secondary: {sec}\n"
        f"  radius:    {rad}"
    )


# ── Return-type structure ─────────────────────────────────────────────────────

def test_result_keys(flat_lc):
    oe = odd_even_depth_test(flat_lc, CATALOG_PERIOD, CATALOG_T0, CATALOG_DURATION)
    sec = secondary_eclipse_test(flat_lc, CATALOG_PERIOD, CATALOG_T0, CATALOG_DURATION)
    shape = shape_metric(flat_lc, CATALOG_PERIOD, CATALOG_T0, CATALOG_DURATION)
    rad = radius_sanity(CATALOG_PRAD)

    assert set(oe.keys()) == {"odd_depth", "even_depth", "depth_ratio", "flag"}
    assert set(sec.keys()) == {"secondary_depth", "significance", "flag"}
    assert set(shape.keys()) == {"vshape_ratio", "flag"}
    assert set(rad.keys()) == {"planet_radius_rearth", "planet_radius_rjup", "flag"}
