import numpy as np
import pytest
from lightkurve import LightCurve

from tools.transit_fit import fit_transit

# K00001.01 / TrES-2b catalog values
CATALOG_PERIOD = 2.470613       # days
CATALOG_T0 = 122.763305         # BKJD
CATALOG_DURATION = 1.74319 / 24.0  # hours → days
CATALOG_PRAD = 13.04            # Earth radii (koi_prad)
# Stellar radius derived from catalog depth + koi_prad (see tools/transit_fit.py header)
STELLAR_RADIUS_RSUN = 1.003


@pytest.fixture(scope="module")
def fit_result(flat_lc):
    return fit_transit(
        flat_lc,
        period=CATALOG_PERIOD,
        t0=CATALOG_T0,
        duration=CATALOG_DURATION,
        stellar_radius_rsun=STELLAR_RADIUS_RSUN,
    )


def test_fit_result_keys(fit_result):
    expected = {
        "converged", "rp_rs", "a_rs", "inc", "impact_parameter",
        "planet_radius_rearth", "reduced_chi2", "vshape_ratio",
    }
    assert set(fit_result.keys()) == expected


def test_fit_converges(fit_result):
    assert fit_result["converged"], (
        f"Fit did not converge on confirmed planet TrES-2b; result: {fit_result}"
    )


def test_fit_planet_radius_sensible(fit_result):
    """Fitted planet radius should be within 30% of the catalog koi_prad."""
    r = fit_result["planet_radius_rearth"]
    assert r is not None
    tol = 0.30
    lo, hi = CATALOG_PRAD * (1 - tol), CATALOG_PRAD * (1 + tol)
    assert lo < r < hi, (
        f"Fitted planet radius {r:.2f} R_Earth is outside ±30% of "
        f"catalog {CATALOG_PRAD} R_Earth  [{lo:.1f}, {hi:.1f}]"
    )


def test_fit_vshape_ratio_range(fit_result):
    """vshape_ratio must lie in [0, 1]; values near 1 indicate a grazing/V-shape transit."""
    v = fit_result["vshape_ratio"]
    assert 0.0 <= v <= 1.0, f"vshape_ratio {v:.3f} is outside valid range [0, 1]"


def test_noise_returns_not_converged():
    """Pure Gaussian noise with no transit should never raise and must return converged=False."""
    rng = np.random.default_rng(42)
    n = 1000
    t = np.linspace(0.0, 80.0, n)
    f = rng.normal(1.0, 1e-3, n).astype(np.float64)
    noise_lc = LightCurve(time=t, flux=f)

    result = fit_transit(noise_lc, period=5.0, t0=2.5, duration=0.1)

    assert not result["converged"], (
        f"Expected converged=False for pure noise input, got: {result}"
    )
