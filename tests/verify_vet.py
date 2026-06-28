#!/usr/bin/env python3
"""Verify vet_target on one confirmed planet and one known false positive.

Confirmed planet : KIC 11446443 — TrES-2b (P ≈ 2.47 d, Rp ≈ 13 R⊕)
False positive   : KIC 3547091  — Kepler EB Catalog contact binary (P ≈ 0.79 d)

The contact binary should trigger at least one of:
  secondary_eclipse.significance > 3.0  (other eclipse sits at phase 0.5)
  shape.vshape_ratio > 0.4              (V-shaped contact-binary profile)
  radius_sanity.planet_radius_rjup > 2  (stellar companion implied radius)

If the FP target doesn't clearly trigger diagnostics, substitute any KIC from
the Kepler Eclipsing Binary Catalog (https://keplerebs.villanova.edu/) with
period 1–5 days so the BLS search (min=0.5, max=30 d) will find it cleanly.

Run:
    python -m tests.verify_vet          # from repo root
    python tests/verify_vet.py          # or directly
"""
import json
import sys

sys.path.insert(0, ".")

from agent.loop import vet_target

# ── Target configuration ──────────────────────────────────────────────────────

CONFIRMED_KEPID = 11446443  # TrES-2b — confirmed hot Jupiter
FP_KEPID        =  3547091  # Kepler EB catalog contact binary, P ≈ 0.79 d

TARGETS = [
    {
        "kepid":    CONFIRMED_KEPID,
        "expected": "CONFIRMED",
        "label":    f"TrES-2b (KIC {CONFIRMED_KEPID})",
    },
    {
        "kepid":    FP_KEPID,
        "expected": "FALSE POSITIVE",
        "label":    f"EB (KIC {FP_KEPID})",
    },
]

REQUIRED_TOOLS = [
    "fetch_lightcurve",
    "detrend",
    "bls_search",
    "fit_transit",
    "compute_diagnostics",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _collect_tools(messages: list) -> list[str]:
    """Return the ordered list of tool names called across all assistant turns."""
    called = []
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        for block in msg["content"]:
            if hasattr(block, "type") and block.type == "tool_use":
                called.append(block.name)
    return called


def _first_assistant_text(messages: list) -> str:
    """Return the text of the very first text block in the first assistant turn."""
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        for block in msg["content"]:
            if hasattr(block, "type") and block.type == "text":
                return block.text
        break
    return ""


def _print_trace(messages: list) -> None:
    """Pretty-print the full conversation trace."""
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[turn {i}] {role}")

        if isinstance(content, list):
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text = block.text
                        snippet = text[:800] + ("…" if len(text) > 800 else "")
                        print(f"  TEXT: {snippet}")
                    elif block.type == "tool_use":
                        args_str = json.dumps(block.input)
                        args_snippet = args_str[:200] + ("…" if len(args_str) > 200 else "")
                        print(f"  TOOL_USE: {block.name}({args_snippet})")
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    res = block.get("content", "")
                    snippet = res[:400] + ("…" if len(res) > 400 else "")
                    print(f"  TOOL_RESULT: {snippet}")
        elif isinstance(content, str):
            print(f"  {content[:400]}")


# ── Assessment ────────────────────────────────────────────────────────────────


def _check(label: str, passed: bool, detail: str = "") -> bool:
    icon = "✓ PASS" if passed else "✗ FAIL"
    line = f"  {icon}  {label}"
    if detail:
        line += f"  [{detail}]"
    print(line)
    return passed


def assess(target: dict, result: dict) -> bool:
    """Print per-check results and return True iff all checks pass."""
    verdict  = result["verdict"]
    messages = result["messages"]
    expected = target["expected"]
    all_ok   = True

    # ── 1. Disposition ────────────────────────────────────────────────────────
    disposition = (verdict or {}).get("disposition", "<no verdict>")
    ok = _check(
        "Correct disposition",
        disposition == expected,
        f"got {disposition!r}, expected {expected!r}",
    )
    all_ok = all_ok and ok

    # ── 2. All required tools called ──────────────────────────────────────────
    tools_called = _collect_tools(messages)
    missing      = [t for t in REQUIRED_TOOLS if t not in tools_called]
    ok = _check(
        "All 5 required tools called",
        not missing,
        f"missing: {missing}" if missing else f"called: {tools_called}",
    )
    all_ok = all_ok and ok

    # ── 3. Sensible tool order (fetch → detrend before anything else) ─────────
    order_ok = True
    if tools_called:
        if tools_called[0] != "fetch_lightcurve":
            order_ok = False
        elif len(tools_called) > 1 and tools_called[1] != "detrend":
            order_ok = False
    ok = _check(
        "Pipeline starts fetch_lightcurve → detrend",
        order_ok,
        f"first two calls: {tools_called[:2]}",
    )
    all_ok = all_ok and ok

    # ── 4. Anti-contamination: no verdict words before first tool result ───────
    # The first assistant text block should not contain a conclusion.
    # The model is allowed to say what it WILL do, not what the answer IS.
    contamination_words = (
        "confirmed planet",
        "false positive",
        "is a planet",
        "is an eclipsing",
        "is tres-2",
        "tres-2b",
    )
    first_text = _first_assistant_text(messages).lower()
    contaminated = any(w in first_text for w in contamination_words)
    ok = _check(
        "No pre-tool verdict contamination",
        not contaminated,
        "first assistant text cited disposition before tools" if contaminated else "",
    )
    all_ok = all_ok and ok

    # ── 5. Verdict cites actual numbers ───────────────────────────────────────
    if verdict:
        evidence    = verdict.get("evidence", {})
        has_numbers = any(
            isinstance(v, (int, float)) and v != 0.0
            for v in evidence.values()
            if v is not None
        )
        ok = _check(
            "Verdict evidence contains non-zero numbers",
            has_numbers,
            str(evidence),
        )
        all_ok = all_ok and ok
    else:
        _check("Verdict was extracted", False, "no JSON verdict found in final response")
        all_ok = False

    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    overall = True

    for target in TARGETS:
        kepid = target["kepid"]
        label = target["label"]

        print(f"\n{'='*68}")
        print(f"  Target  : {label}")
        print(f"  Expected: {target['expected']}")
        print("=" * 68)

        result  = vet_target(kepid)
        verdict = result["verdict"]

        _print_trace(result["messages"])

        print("\n--- Verdict ---")
        if verdict:
            print(json.dumps(verdict, indent=2))
        else:
            print("(no verdict extracted)")

        print("\n--- Assessment ---")
        passed  = assess(target, result)
        overall = overall and passed

    print(f"\n{'='*68}")
    print(f"  Overall: {'ALL PASS ✓' if overall else 'SOME FAILURES ✗'}")
    print("=" * 68)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
