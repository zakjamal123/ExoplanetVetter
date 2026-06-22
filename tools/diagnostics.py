"""Binary-star false-positive diagnostic tests.

Each function returns a TypedDict with numeric metrics and a boolean flag.
All phase arithmetic folds on the *reported* period (which may be half the
true EB period — that is exactly what several tests exploit).
"""
from typing import TypedDict

import numpy as np
from lightkurve import LightCurve

_R_JUP_IN_REARTH = 11.21


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class OddEvenResult(TypedDict):
    odd_depth: float    # mean fractional depth of odd-numbered transits
    even_depth: float   # mean fractional depth of even-numbered transits
    depth_ratio: float  # |odd - even| / mean_depth; large → suspicious
    flag: bool          # True when ratio > 0.1 (10 % relative mismatch)


class SecondaryEclipseResult(TypedDict):
    secondary_depth: float  # fractional dip at phase 0.5 (positive = below continuum)
    significance: float     # sigma above out-of-transit scatter
    flag: bool              # True when significance > 3


class ShapeMetricResult(TypedDict):
    vshape_ratio: float  # 0 = flat-bottomed U-shape; 1 = pointed V-shape
    flag: bool           # True when ratio > 0.4 (likely grazing EB)


class RadiusSanityResult(TypedDict):
    planet_radius_rearth: float
    planet_radius_rjup: float
    flag: bool  # True when radius > 2 R_Jup (likely stellar companion)


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------

def odd_even_depth_test(
    lc: LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> OddEvenResult:
    """Compare depths of odd- vs even-numbered transits.

    A systematic depth difference implies the true period is double the
    reported one (primary + secondary eclipses of an EB alternating).
    """
    t = lc.time.value
    f = lc.flux.value
    h = duration / 2.0

    n_min = int(np.ceil((t.min() - t0) / period))
    n_max = int(np.floor((t.max() - t0) / period))
    transit_times = [t0 + n * period for n in range(n_min, n_max + 1)]

    odd_depths, even_depths = [], []
    for rank, tc in enumerate(transit_times, start=1):
        mask = np.abs(t - tc) < h
        if mask.sum() < 3:
            continue
        depth = 1.0 - float(np.median(f[mask]))
        (odd_depths if rank % 2 == 1 else even_depths).append(depth)

    if len(odd_depths) < 2 or len(even_depths) < 2:
        return OddEvenResult(
            odd_depth=float("nan"), even_depth=float("nan"),
            depth_ratio=float("nan"), flag=False,
        )

    odd_mean = float(np.mean(odd_depths))
    even_mean = float(np.mean(even_depths))
    mean_depth = (odd_mean + even_mean) / 2.0

    if mean_depth <= 0.0:
        return OddEvenResult(
            odd_depth=odd_mean, even_depth=even_mean,
            depth_ratio=0.0, flag=False,
        )

    depth_ratio = float(abs(odd_mean - even_mean) / mean_depth)
    return OddEvenResult(
        odd_depth=odd_mean,
        even_depth=even_mean,
        depth_ratio=depth_ratio,
        flag=depth_ratio > 0.1,
    )


def secondary_eclipse_test(
    lc: LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> SecondaryEclipseResult:
    """Search for a significant dip at phase ≈ 0.5.

    If the reported period is already P_true / 2, each alternate 'transit'
    is actually the secondary eclipse, so folding at the reported period
    places it at phase 0.5.
    """
    t = lc.time.value
    f = lc.flux.value

    phase = ((t - t0) % period) / period          # [0, 1)
    h_frac = (duration / 2.0) / period            # half-duration in phase units

    # Exclude the primary and secondary transit windows from OOT estimate
    oot_mask = (
        ((phase > 2.0 * h_frac) & (phase < 0.5 - 2.0 * h_frac)) |
        ((phase > 0.5 + 2.0 * h_frac) & (phase < 1.0 - 2.0 * h_frac))
    )
    sec_mask = np.abs(phase - 0.5) < h_frac

    _fail = SecondaryEclipseResult(secondary_depth=0.0, significance=0.0, flag=False)
    if oot_mask.sum() < 10 or sec_mask.sum() < 3:
        return _fail

    oot_med = float(np.median(f[oot_mask]))
    oot_std = float(np.std(f[oot_mask]))
    sec_med = float(np.median(f[sec_mask]))

    secondary_depth = oot_med - sec_med   # positive = dip below continuum
    significance = (
        float(secondary_depth / (oot_std / np.sqrt(sec_mask.sum())))
        if oot_std > 0.0 else 0.0
    )
    return SecondaryEclipseResult(
        secondary_depth=float(secondary_depth),
        significance=float(significance),
        flag=significance > 3.0,
    )


def shape_metric(
    lc: LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> ShapeMetricResult:
    """Quantify transit shape from the folded light curve.

    Compares the flux depth in the mid-ingress zone (|phase| ≈ h/2) to the
    depth at the transit core (|phase| < h/3).

    U-shape (flat bottom): ingress is fast, so the mid-ingress zone is still
    near full depth → vshape_ratio ≈ 0.
    V-shape (grazing/no flat bottom): depth falls linearly across the whole
    transit → mid-ingress depth ≈ 0.5 × core depth → vshape_ratio ≈ 0.5.
    """
    t = lc.time.value
    f = lc.flux.value

    h = duration / 2.0
    # Phase centred on t0 in [-P/2, P/2]
    phase = ((t - t0 + period / 2.0) % period) - period / 2.0

    core_mask = np.abs(phase) < h / 3.0
    wing_mask = (np.abs(phase) > h / 3.0) & (np.abs(phase) < 2.0 * h / 3.0)
    oot_mask = (np.abs(phase) > duration) & (np.abs(phase) < 3.0 * duration)

    _fail = ShapeMetricResult(vshape_ratio=float("nan"), flag=False)
    if core_mask.sum() < 3 or wing_mask.sum() < 3 or oot_mask.sum() < 5:
        return _fail

    oot_level = float(np.median(f[oot_mask]))
    core_depth = oot_level - float(np.median(f[core_mask]))
    wing_depth = oot_level - float(np.median(f[wing_mask]))

    if core_depth <= 0.0:
        return _fail

    # High ratio → wing already at full depth → U-shape (planet-like)
    # Low  ratio → wing much shallower than core → V-shape (EB-like)
    vshape_ratio = float(np.clip(1.0 - wing_depth / core_depth, 0.0, 1.0))
    return ShapeMetricResult(
        vshape_ratio=vshape_ratio,
        flag=vshape_ratio > 0.4,
    )


def radius_sanity(planet_radius_rearth: float) -> RadiusSanityResult:
    """Flag implied radii above 2 R_Jup as likely stellar companions."""
    rjup = planet_radius_rearth / _R_JUP_IN_REARTH
    return RadiusSanityResult(
        planet_radius_rearth=float(planet_radius_rearth),
        planet_radius_rjup=float(rjup),
        flag=rjup > 2.0,
    )
