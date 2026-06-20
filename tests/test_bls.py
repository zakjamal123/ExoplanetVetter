import pandas as pd
import pytest

from tools.bls import bls_search
from tools.detrend import detrend
from tools.fetch import fetch_lightcurve

# K00001.01 / TrES-2b — the Task 1.2 sanity-check target
KEPID = 11446443
CATALOG_PERIOD = 2.470613  # days, from data/koi_table.parquet

# Harmonic ratios to check: BLS can lock onto P/2 or 2P instead of P
_HARMONICS = {
    "half (P/2)": 0.5,
    "double (2P)": 2.0,
}
_HARMONIC_TOL = 0.01  # 1% — same tolerance used for the primary match


def _harmonic_locked(recovered: float, catalog: float) -> str | None:
    """Return a description string if `recovered` is near a harmonic of `catalog`."""
    for name, ratio in _HARMONICS.items():
        if abs(recovered - ratio * catalog) / (ratio * catalog) < _HARMONIC_TOL:
            return name
    return None


@pytest.fixture(scope="module")
def bls_result(flat_lc):
    return bls_search(flat_lc, min_period=1.0, max_period=6.0)


@pytest.fixture(scope="module")
def flat_lc():
    lc = fetch_lightcurve(KEPID)
    return detrend(lc)


def test_bls_period_within_1pct(bls_result):
    recovered = bls_result["period"]
    error_pct = abs(recovered - CATALOG_PERIOD) / CATALOG_PERIOD * 100
    assert error_pct < 1.0, (
        f"BLS period {recovered:.6f} d is {error_pct:.3f}% from "
        f"catalog {CATALOG_PERIOD:.6f} d (tolerance 1%)"
    )


def test_no_harmonic_lock(bls_result):
    """Fail with an informative message if BLS landed on P/2 or 2P."""
    recovered = bls_result["period"]
    harmonic = _harmonic_locked(recovered, CATALOG_PERIOD)
    assert harmonic is None, (
        f"BLS locked onto the {harmonic} harmonic: recovered {recovered:.6f} d "
        f"instead of catalog {CATALOG_PERIOD:.6f} d. "
        "Tighten the period grid around the catalog value or increase n_periods."
    )


def test_bls_result_keys(bls_result):
    assert set(bls_result.keys()) == {"period", "t0", "duration", "depth", "power"}
    assert all(isinstance(v, float) for v in bls_result.values())


def test_catalog_period_matches_koi_table():
    """Sanity-check that the hardcoded period still matches the cached table."""
    df = pd.read_parquet("data/koi_table.parquet")
    row = df[df["kepid"] == KEPID].iloc[0]
    assert abs(row["koi_period"] - CATALOG_PERIOD) < 1e-4
