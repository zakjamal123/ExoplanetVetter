"""Tool schemas and dispatcher for the exoplanet-vetting agent.

LightCurve objects cannot cross the JSON boundary, so each tool that produces
one stores it in a module-level cache and returns a string handle.  Downstream
tools accept that handle and resolve it back to the object.

Pipeline order:
  fetch_lightcurve  → raw_lc_handle
  detrend           → flat_lc_handle
  bls_search        → period / t0 / duration / depth / power
  fit_transit       → rp_rs / a_rs / inc / … / planet_radius_rearth
  compute_diagnostics → odd_even / secondary_eclipse / shape / radius_sanity
"""
import json
import math
from typing import Any

from lightkurve import LightCurve

from tools.bls import bls_search
from tools.detrend import detrend
from tools.diagnostics import (
    odd_even_depth_test,
    radius_sanity,
    secondary_eclipse_test,
    shape_metric,
)
from tools.fetch import fetch_lightcurve
from tools.transit_fit import fit_transit

# ── In-process light-curve cache ─────────────────────────────────────────────

_LC_CACHE: dict[str, LightCurve] = {}


def _get_lc(handle: str) -> LightCurve:
    if handle not in _LC_CACHE:
        raise KeyError(
            f"Light-curve handle {handle!r} not found in cache. "
            "Call fetch_lightcurve (and detrend) first."
        )
    return _LC_CACHE[handle]


# ── JSON-safety helper ────────────────────────────────────────────────────────

def _to_json_safe(obj: Any) -> Any:
    """Recursively replace NaN / ±Inf with None so json.dumps never raises."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, bool):      # bool before int — bool is a subclass of int
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (int, str, type(None))):
        return obj
    return obj


# ── Tool schemas (Anthropic API format) ──────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "fetch_lightcurve",
        "description": (
            "Download and stitch all available Kepler long-cadence quarters for a "
            "KIC target from MAST. Returns a raw light-curve handle to pass to detrend."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kepid": {
                    "type": "integer",
                    "description": "Kepler Input Catalog identifier (e.g. 11446443 for TrES-2b).",
                },
            },
            "required": ["kepid"],
        },
    },
    {
        "name": "detrend",
        "description": (
            "Flatten a raw light curve with a Savitzky-Golay filter and sigma-clip "
            "upward outliers (cosmic rays / flares). Returns a flat light-curve handle "
            "suitable for BLS, transit fitting, and diagnostics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "raw_lc_handle": {
                    "type": "string",
                    "description": "Handle returned by fetch_lightcurve.",
                },
                "window_length": {
                    "type": "integer",
                    "description": (
                        "Savitzky-Golay filter window in cadences (must be odd). "
                        "Default 401 ≈ 8 days, appropriate for hot-Jupiter periods."
                    ),
                    "default": 401,
                },
            },
            "required": ["raw_lc_handle"],
        },
    },
    {
        "name": "bls_search",
        "description": (
            "Run Box Least Squares on a detrended light curve to find the best-fit "
            "periodic transit signal. Returns period, epoch (t0), duration, fractional "
            "depth, and peak BLS power."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "flat_lc_handle": {
                    "type": "string",
                    "description": "Handle returned by detrend.",
                },
                "min_period": {
                    "type": "number",
                    "description": "Minimum trial period in days.",
                },
                "max_period": {
                    "type": "number",
                    "description": "Maximum trial period in days.",
                },
                "n_periods": {
                    "type": "integer",
                    "description": "Number of evenly-spaced period grid points. Default 2000.",
                    "default": 2000,
                },
            },
            "required": ["flat_lc_handle", "min_period", "max_period"],
        },
    },
    {
        "name": "fit_transit",
        "description": (
            "Fit a batman transit model (free params: Rp/Rs, a/Rs, inclination; "
            "eccentricity fixed to 0) via L-BFGS-B. Returns fitted parameters, "
            "reduced chi-squared, V-shape ratio, and — when stellar_radius_rsun is "
            "provided — the implied planet radius in Earth radii. "
            "Returns converged=false instead of raising on optimizer failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "flat_lc_handle": {
                    "type": "string",
                    "description": "Handle returned by detrend.",
                },
                "period": {
                    "type": "number",
                    "description": "Orbital period in days (e.g. from bls_search).",
                },
                "t0": {
                    "type": "number",
                    "description": "Mid-transit reference time in BKJD (e.g. from bls_search).",
                },
                "duration": {
                    "type": "number",
                    "description": "Approximate transit duration in days (e.g. from bls_search).",
                },
                "stellar_radius_rsun": {
                    "type": "number",
                    "description": (
                        "Host-star radius in solar radii. When provided, "
                        "planet_radius_rearth is populated in the output."
                    ),
                },
            },
            "required": ["flat_lc_handle", "period", "t0", "duration"],
        },
    },
    {
        "name": "compute_diagnostics",
        "description": (
            "Run all four binary-star false-positive diagnostic tests on a detrended "
            "light curve: odd/even transit depth comparison, secondary eclipse search "
            "at phase 0.5, V-shape metric, and (optionally) planet radius sanity check. "
            "any_flag=true means at least one test is suspicious."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "flat_lc_handle": {
                    "type": "string",
                    "description": "Handle returned by detrend.",
                },
                "period": {
                    "type": "number",
                    "description": "Orbital period in days.",
                },
                "t0": {
                    "type": "number",
                    "description": "Mid-transit reference time in BKJD.",
                },
                "duration": {
                    "type": "number",
                    "description": "Transit duration in days.",
                },
                "planet_radius_rearth": {
                    "type": "number",
                    "description": (
                        "Implied planet radius in Earth radii from fit_transit. "
                        "When provided, enables the radius_sanity test (flags > 2 R_Jup)."
                    ),
                },
            },
            "required": ["flat_lc_handle", "period", "t0", "duration"],
        },
    },
]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def run_tool(name: str, args: dict) -> dict:
    """Map tool name + JSON-safe args to the real Python function.

    Returns a JSON-serializable dict (NaN / ±Inf replaced with null).
    Raises ValueError for unknown tool names.
    Raises KeyError if a required light-curve handle is not in the cache.
    """
    match name:

        case "fetch_lightcurve":
            kepid = int(args["kepid"])
            lc = fetch_lightcurve(kepid)
            handle = f"raw:{kepid}"
            _LC_CACHE[handle] = lc
            t = lc.time.value
            return _to_json_safe({
                "raw_lc_handle": handle,
                "n_cadences": int(len(t)),
                "time_span_days": float(t[-1] - t[0]),
            })

        case "detrend":
            raw_handle = str(args["raw_lc_handle"])
            window_length = int(args.get("window_length", 401))
            flat = detrend(_get_lc(raw_handle), window_length=window_length)
            # Derive a stable flat handle from the raw handle ("raw:X" → "flat:X")
            suffix = raw_handle.split(":", 1)[1] if ":" in raw_handle else raw_handle
            flat_handle = f"flat:{suffix}"
            _LC_CACHE[flat_handle] = flat
            return _to_json_safe({
                "flat_lc_handle": flat_handle,
                "n_cadences": int(len(flat.time)),
            })

        case "bls_search":
            flat = _get_lc(str(args["flat_lc_handle"]))
            result = bls_search(
                flat,
                min_period=float(args["min_period"]),
                max_period=float(args["max_period"]),
                n_periods=int(args.get("n_periods", 2000)),
            )
            return _to_json_safe(dict(result))

        case "fit_transit":
            flat = _get_lc(str(args["flat_lc_handle"]))
            stellar_r = args.get("stellar_radius_rsun")
            result = fit_transit(
                flat,
                period=float(args["period"]),
                t0=float(args["t0"]),
                duration=float(args["duration"]),
                stellar_radius_rsun=float(stellar_r) if stellar_r is not None else None,
            )
            return _to_json_safe(dict(result))

        case "compute_diagnostics":
            flat = _get_lc(str(args["flat_lc_handle"]))
            period = float(args["period"])
            t0 = float(args["t0"])
            duration = float(args["duration"])

            oe = odd_even_depth_test(flat, period, t0, duration)
            sec = secondary_eclipse_test(flat, period, t0, duration)
            shape = shape_metric(flat, period, t0, duration)

            rad_raw = args.get("planet_radius_rearth")
            rad = radius_sanity(float(rad_raw)) if rad_raw is not None else None

            any_flag = (
                oe["flag"]
                or sec["flag"]
                or shape["flag"]
                or (rad["flag"] if rad is not None else False)
            )
            return _to_json_safe({
                "odd_even": dict(oe),
                "secondary_eclipse": dict(sec),
                "shape": dict(shape),
                "radius_sanity": dict(rad) if rad is not None else None,
                "any_flag": bool(any_flag),
            })

        case _:
            raise ValueError(f"Unknown tool: {name!r}")

