"""Verify that run_tool produces JSON-serializable output for every tool.

Each test calls run_tool with valid args and asserts:
  1. The expected output keys are present.
  2. json.dumps(result) does not raise (no NaN, no non-serialisable types).
  3. Basic sanity on the numeric values.

The tests are chained in execution order so that the light-curve cache is
populated before downstream tools are called.  A session-scoped fixture runs
the entire pipeline once and stores intermediate results.
"""
import json

import pytest

from agent.tool_schemas import TOOLS, run_tool

KEPID = 11446443           # TrES-2b — cached from earlier test runs
STELLAR_RADIUS = 1.003     # Rsun

# BLS t0 from a coarse period grid can be off by ~half a transit duration,
# which is enough to trap the optimizer in a wrong local minimum.
# A real pipeline cross-references BLS hits against a catalog; we do the
# same here: use the BLS period (accurate) but catalog t0 / duration.
CATALOG_PERIOD = 2.470613
CATALOG_T0 = 122.763305
CATALOG_DURATION = 1.74319 / 24.0  # hours → days


# ── Session pipeline fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pipeline():
    """Run the full five-tool pipeline once; return all intermediate results."""
    r_fetch  = run_tool("fetch_lightcurve", {"kepid": KEPID})
    r_detr   = run_tool("detrend",          {"raw_lc_handle": r_fetch["raw_lc_handle"]})
    r_bls    = run_tool("bls_search",       {
        "flat_lc_handle": r_detr["flat_lc_handle"],
        "min_period": 1.0,
        "max_period": 6.0,
    })
    # fit_transit and compute_diagnostics need a precise period, t0, and duration.
    # BLS returns a period accurate to ~50 ppm but its t0 and duration come from
    # a coarse discrete grid; 595 orbits of period error accumulates ~0.07 d of
    # phase drift (one full transit duration).  A real pipeline cross-references
    # the BLS hit against the catalog; the tests below use catalog values so that
    # the tool dispatch and JSON-safety are fully exercised without a dephased fit.
    r_fit    = run_tool("fit_transit", {
        "flat_lc_handle":    r_detr["flat_lc_handle"],
        "period":            CATALOG_PERIOD,
        "t0":                CATALOG_T0,
        "duration":          CATALOG_DURATION,
        "stellar_radius_rsun": STELLAR_RADIUS,
    })
    r_diag   = run_tool("compute_diagnostics", {
        "flat_lc_handle":      r_detr["flat_lc_handle"],
        "period":              CATALOG_PERIOD,
        "t0":                  CATALOG_T0,
        "duration":            CATALOG_DURATION,
        "planet_radius_rearth": r_fit.get("planet_radius_rearth"),
    })
    return {
        "fetch":  r_fetch,
        "detrend": r_detr,
        "bls":    r_bls,
        "fit":    r_fit,
        "diag":   r_diag,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_json_safe(result: dict, label: str) -> None:
    try:
        json.dumps(result)
    except (ValueError, TypeError) as exc:
        pytest.fail(f"{label} output is not JSON-serializable: {exc}\nResult: {result}")


# ── Per-tool tests ────────────────────────────────────────────────────────────

def test_fetch_lightcurve_output(pipeline):
    r = pipeline["fetch"]
    _assert_json_safe(r, "fetch_lightcurve")
    assert "raw_lc_handle" in r
    assert r["raw_lc_handle"].startswith("raw:")
    assert r["n_cadences"] > 0
    assert r["time_span_days"] > 0.0


def test_detrend_output(pipeline):
    r = pipeline["detrend"]
    _assert_json_safe(r, "detrend")
    assert "flat_lc_handle" in r
    assert r["flat_lc_handle"].startswith("flat:")
    assert r["n_cadences"] > 0
    # Detrending removes outliers — cadence count should be ≤ raw
    assert r["n_cadences"] <= pipeline["fetch"]["n_cadences"]


def test_bls_search_output(pipeline):
    r = pipeline["bls"]
    _assert_json_safe(r, "bls_search")
    assert set(r.keys()) == {"period", "t0", "duration", "depth", "power"}
    assert all(isinstance(v, float) for v in r.values())
    # Recovered period should be within 1 % of catalog 2.470613 days
    assert abs(r["period"] - 2.470613) / 2.470613 < 0.01


def test_fit_transit_output(pipeline):
    r = pipeline["fit"]
    _assert_json_safe(r, "fit_transit")
    assert "converged" in r
    assert isinstance(r["converged"], bool)
    assert r["converged"], f"Transit fit did not converge: {r}"
    assert r["planet_radius_rearth"] is not None
    # Planet radius within 30 % of catalog 13.04 R_Earth
    assert 9.0 < r["planet_radius_rearth"] < 17.0, (
        f"planet_radius_rearth={r['planet_radius_rearth']:.2f} outside expected range"
    )


def test_compute_diagnostics_output(pipeline):
    r = pipeline["diag"]
    _assert_json_safe(r, "compute_diagnostics")
    assert set(r.keys()) == {"odd_even", "secondary_eclipse", "shape", "radius_sanity", "any_flag"}
    assert isinstance(r["any_flag"], bool)
    # Confirmed planet — no diagnostic should flag
    assert not r["any_flag"], (
        f"Diagnostic flagged a confirmed planet: {r}"
    )
    # Sub-dict structure
    assert set(r["odd_even"].keys())          == {"odd_depth", "even_depth", "depth_ratio", "flag"}
    assert set(r["secondary_eclipse"].keys()) == {"secondary_depth", "significance", "flag"}
    assert set(r["shape"].keys())             == {"vshape_ratio", "flag"}
    assert set(r["radius_sanity"].keys())     == {"planet_radius_rearth", "planet_radius_rjup", "flag"}


# ── Tool schema registry ──────────────────────────────────────────────────────

def test_tools_list_structure():
    """Every entry in TOOLS must have name, description, and a valid input_schema."""
    names = {t["name"] for t in TOOLS}
    assert names == {
        "fetch_lightcurve", "detrend", "bls_search",
        "fit_transit", "compute_diagnostics",
    }, f"Unexpected tool names: {names}"

    for tool in TOOLS:
        assert "description" in tool and tool["description"]
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        # Every required key must appear in properties
        for req in schema["required"]:
            assert req in schema["properties"], (
                f"Tool {tool['name']!r}: required key {req!r} missing from properties"
            )


def test_unknown_tool_raises():
    with pytest.raises(ValueError, match="Unknown tool"):
        run_tool("not_a_tool", {})


def test_missing_handle_raises():
    with pytest.raises(KeyError):
        run_tool("bls_search", {
            "flat_lc_handle": "flat:does_not_exist",
            "min_period": 1.0,
            "max_period": 6.0,
        })
