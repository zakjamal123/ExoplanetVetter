"""Agentic tool-use loop for exoplanet transit vetting."""
import json
import re
from typing import Any

import anthropic

from agent.prompts import SYSTEM_PROMPT
from agent.tool_schemas import TOOLS, run_tool

_MODEL = "claude-sonnet-4-6"
_MAX_STEPS = 20


def _call_tool(block: Any) -> dict:
    try:
        return run_tool(block.name, block.input)
    except Exception as exc:
        return {"error": str(exc)}


def _extract_verdict(text: str) -> dict | None:
    """Parse the JSON verdict from the model's final text block."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: bare JSON object that contains the "disposition" key
    m = re.search(r"\{[^{}]*\"disposition\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def vet_target(kepid: int, max_steps: int = _MAX_STEPS) -> dict[str, Any]:
    """Run a Claude tool-use loop to vet a Kepler target.

    The loop sends the initial user message, executes any tool_use blocks
    returned by the model, appends tool_result messages, and repeats until
    the model emits a verdict (stop_reason == "end_turn") or max_steps is hit.

    Returns
    -------
    dict with keys:
        verdict  — parsed JSON verdict dict, or None if the cap was hit
        messages — full conversation history (assistant content blocks intact)
    """
    client = anthropic.Anthropic()

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Vet Kepler Input Catalog target KIC {kepid}. "
                "Follow the mandatory workflow, call all required tools, "
                "and emit your verdict."
            ),
        }
    ]

    verdict: dict | None = None

    for _ in range(max_steps):
        response = client.messages.create(
            model=_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append the assistant turn; the SDK preserves tool_use block objects
        # so subsequent API calls can reference their IDs correctly.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    v = _extract_verdict(block.text)
                    if v is not None:
                        verdict = v
                        break
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _call_tool(block)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

    return {"verdict": verdict, "messages": messages}
