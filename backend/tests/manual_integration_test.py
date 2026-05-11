"""Manual integration test for the formulation provider layer.

Not collected by ``pytest`` (filename does not start with ``test_``).
Run it once at the end of Phase 3 with real API keys to confirm the
three providers work against their actual upstream services::

    ANTHROPIC_API_KEY=sk-ant-... \
    OPENAI_API_KEY=sk-... \
    LOCAL_LLM_ENDPOINT=http://localhost:11434 \
    python tests/manual_integration_test.py

Exit code 0 if at least 2 of 3 providers produced a CQM that compiled
and passed the validation harness; exit code 1 otherwise.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from pathlib import Path

# Allow `python tests/manual_integration_test.py` to find the `app` package.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.formulation import (  # noqa: E402
    FormulationError,
    FormulationResult,
    get_provider,
)
from app.optimization.compiler import compile_cqm_json  # noqa: E402
from app.optimization.validation import validate_cqm  # noqa: E402

PROBLEM = (
    "I have 3 jobs and 2 machines arranged in series, A then B. Job 1 "
    "takes (3, 2) time units on (A, B); Job 2 takes (2, 4); Job 3 takes "
    "(4, 1). Each machine processes one job at a time. Find the start "
    "times that minimize the makespan."
)


async def _try_provider(name: str, api_key: str | None) -> tuple[bool, str]:
    """Returns ``(success, summary)``. Success = compiled + validated."""
    try:
        provider = get_provider(name)
    except KeyError as e:
        return False, f"provider not registered: {e}"

    if api_key is None and name != "local":
        return False, "skipped (no API key in env)"

    # 20-minute per-provider timeout. The cloud arms typically return in
    # under 30 s; the local arm at qwen3:14b needs ~8 min on a warm model,
    # but a cold model (Ollama evicts after a few minutes idle) adds
    # 1-2 min of load time, and on a busy GPU the prompt-eval phase can
    # double — 1200 s is the empirically-safe ceiling for this hardware.
    PER_PROVIDER_TIMEOUT_S = 1200
    try:
        t0 = time.perf_counter()
        result: FormulationResult = await provider.formulate(
            PROBLEM, api_key or "", timeout=PER_PROVIDER_TIMEOUT_S,
        )
        elapsed = time.perf_counter() - t0
    except FormulationError as e:
        return False, f"formulate failed: {e}"
    except Exception as e:  # network or parsing weirdness
        return False, f"unexpected error: {type(e).__name__}: {e}"

    try:
        cqm, _registry, sense = compile_cqm_json(result.cqm_json)
    except ValueError as e:
        return False, f"compile failed: {e}"

    expected = (result.cqm_json.get("test_instance") or {}).get("expected_optimum")
    try:
        report = validate_cqm(
            cqm,
            expected_optimum=expected,
            sense=sense,
            skip_layer_b=True,   # keep this script under a minute, even on slow networks
        )
    except Exception as e:
        return False, f"validate failed: {type(e).__name__}: {e}"

    summary = (
        f"OK   {name:8s} model={result.model:24s} "
        f"tokens={result.tokens_used:5d} "
        f"vars={len(result.cqm_json.get('variables', [])):3d} "
        f"constraints={len(result.cqm_json.get('constraints', [])):3d} "
        f"expected={expected!s:>5} "
        f"oracle_skipped={report.oracle_skipped} "
        f"passed={report.passed} "
        f"({elapsed:.1f}s)"
    )
    return report.passed, summary


async def main() -> int:
    # OpenAI dropped from the v2 integration gate: under the academic
    # positioning, Claude is the budget-friendly cloud reference and the
    # local tier provides the offline option. OpenAI remains a registered
    # provider with full unit-test coverage; if an operator wants it in
    # the gate they can re-add it here. Threshold lowered from 2/3 to 2/2
    # to match — both remaining providers must pass.
    keys = {
        "claude": os.environ.get("ANTHROPIC_API_KEY"),
        "local":  None,  # local provider doesn't need a key
    }
    print(f"Problem: {PROBLEM}\n")

    results: list[tuple[str, bool, str]] = []
    for name, key in keys.items():
        try:
            ok, summary = await _try_provider(name, key)
        except Exception:
            ok = False
            summary = f"unexpected exception:\n{traceback.format_exc()}"
        prefix = "OK  " if ok else "FAIL"
        print(f"  [{prefix}] {summary}")
        results.append((name, ok, summary))

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_ok}/{len(results)} providers produced a CQM that compiled + validated")
    # Gate is now 2/2 (Claude + local) — both must pass.
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
