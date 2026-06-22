import warnings
from typing import TypedDict

import batman
import numpy as np
from lightkurve import LightCurve
from scipy.optimize import minimize

_R_SUN_IN_REARTH = 109.076  # IAU 2015 nominal


class FitResult(TypedDict):
    converged: bool
    rp_rs: float
    a_rs: float
    inc: float
    impact_parameter: float
    planet_radius_rearth: float | None
    reduced_chi2: float
    vshape_ratio: float  # (T_ingress+T_egress)/T_total; ~0=flat-bottom U, ~1=V-shape


def _batman_flux(t: np.ndarray, period: float, t0: float,
                 rp_rs: float, a_rs: float, inc: float) -> np.ndarray:
    p = batman.TransitParams()
    p.t0 = t0
    p.per = period
    p.rp = rp_rs
    p.a = a_rs
    p.inc = inc
    p.ecc = 0.0
    p.w = 90.0
    p.u = [0.4, 0.3]
    p.limb_dark = "quadratic"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = batman.TransitModel(p, t)
        return m.light_curve(p)


def _vshape_ratio(rp_rs: float, a_rs: float, inc_deg: float, period: float) -> float:
    """Return (T_ingress + T_egress) / T_14 from transit geometry.

    Uses Seager & Mallen-Ornelas (2003) contact-time formulae.
    Returns 1.0 for grazing or degenerate configurations.
    """
    inc = np.radians(inc_deg)
    b = a_rs * np.cos(inc)
    denom = a_rs * np.sin(inc)
    if denom <= 0.0:
        return 1.0

    arg14 = np.clip((1.0 + rp_rs) ** 2 - b ** 2, 0.0, None)
    arg23 = np.clip((1.0 - rp_rs) ** 2 - b ** 2, 0.0, None)

    T14 = period / np.pi * np.arcsin(np.clip(np.sqrt(arg14) / denom, 0.0, 1.0))
    T23 = period / np.pi * np.arcsin(np.clip(np.sqrt(arg23) / denom, 0.0, 1.0))

    if T14 <= 0.0:
        return 1.0
    return float(np.clip((T14 - T23) / T14, 0.0, 1.0))


def fit_transit(
    lc: LightCurve,
    period: float,
    t0: float,
    duration: float,
    stellar_radius_rsun: float | None = None,
) -> FitResult:
    """Fit a batman transit model to *lc* with fixed period and epoch.

    Free parameters: Rp/Rs, a/Rs, inclination.
    Eccentricity is fixed to 0; limb darkening fixed to quadratic [0.4, 0.3].

    Parameters
    ----------
    lc:
        Detrended, normalised LightCurve.
    period, t0:
        Orbital period (days) and reference mid-transit time (BKJD).
    duration:
        Approximate transit duration in days (used only to mask in-transit
        cadences when estimating out-of-transit scatter).
    stellar_radius_rsun:
        Host-star radius in solar radii.  When provided, ``planet_radius_rearth``
        is populated; otherwise it is None.

    Returns
    -------
    FitResult with ``converged=False`` on optimizer failure or unphysical solution.
    """
    _FAIL: FitResult = FitResult(
        converged=False,
        rp_rs=float("nan"),
        a_rs=float("nan"),
        inc=float("nan"),
        impact_parameter=float("nan"),
        planet_radius_rearth=None,
        reduced_chi2=float("nan"),
        vshape_ratio=float("nan"),
    )

    t = lc.time.value
    f = lc.flux.value

    # Phase relative to transit center, folded into [-P/2, P/2]
    phase = ((t - t0) % period + period / 2.0) % period - period / 2.0
    out_mask = np.abs(phase) > duration
    in_mask = ~out_mask

    if out_mask.sum() < 10:
        return _FAIL

    sigma = float(np.std(f[out_mask]))
    if sigma <= 0.0:
        sigma = 1.0e-6

    n_free = 3

    def cost(x: np.ndarray) -> float:
        rp, a, inc = x
        try:
            flux_model = _batman_flux(t, period, t0, rp, a, inc)
            return float(np.sum(((f - flux_model) / sigma) ** 2))
        except Exception:
            return 1.0e12

    depth_est = max(1.0 - float(np.median(f[in_mask]) if in_mask.sum() > 0 else 1.0), 1e-4)
    rp0 = float(np.clip(np.sqrt(depth_est), 0.02, 0.4))
    x0 = [rp0, 8.0, 87.0]
    bounds = [(0.001, 0.5), (1.5, 100.0), (60.0, 90.0)]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                cost, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7},
            )

        rp_rs = float(result.x[0])
        a_rs = float(result.x[1])
        inc = float(result.x[2])

        # A solution at a parameter bound means no genuine transit was found
        _tol = 5e-3
        at_lb = (rp_rs - 0.001 < _tol) or (a_rs - 1.5 < _tol) or (inc - 60.0 < _tol)
        at_ub = (0.5 - rp_rs < _tol) or (90.0 - inc < _tol)

        converged = bool(result.success) and not at_lb and not at_ub

        b = float(a_rs * np.cos(np.radians(inc)))
        red_chi2 = float(result.fun / max(len(t) - n_free, 1))
        vshape = _vshape_ratio(rp_rs, a_rs, inc, period)

        planet_radius_rearth = (
            float(rp_rs * stellar_radius_rsun * _R_SUN_IN_REARTH)
            if stellar_radius_rsun is not None
            else None
        )

        return FitResult(
            converged=converged,
            rp_rs=rp_rs,
            a_rs=a_rs,
            inc=inc,
            impact_parameter=b,
            planet_radius_rearth=planet_radius_rearth,
            reduced_chi2=red_chi2,
            vshape_ratio=vshape,
        )

    except Exception:
        return _FAIL
