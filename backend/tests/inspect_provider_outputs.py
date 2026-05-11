"""Manual inspection script: dump the actual CQM JSON each provider
produces, plus the raw LLM output, so a reviewer can see "what does
the conversion from natural language to CQM JSON actually look like."

Not a test — does not collect in pytest. Run directly::

    ANTHROPIC_API_KEY=... LOCAL_LLM_MODEL=qwen3:14b \
    python tests/inspect_provider_outputs.py

Outputs:
  /tmp/cira_quantum_<provider>_cqm.json   — the parsed CQM JSON object
  /tmp/cira_quantum_<provider>_raw.txt    — the raw LLM text response
  stdout                                  — a side-by-side summary table
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Allow `python tests/...` to find the `app` package.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.formulation import FormulationError, get_provider  # noqa: E402
from app.optimization.compiler import compile_cqm_json  # noqa: E402

PROBLEM = (
    "I have 3 jobs and 2 machines arranged in series, A then B. Job 1 "
    "takes (3, 2) time units on (A, B); Job 2 takes (2, 4); Job 3 takes "
    "(4, 1). Each machine processes one job at a time. Find the start "
    "times that minimize the makespan."
)

OUT_DIR = Path("/tmp")


async def _try_one(name: str, api_key: str | None, timeout: int) -> dict:
    """Run one provider end-to-end. Returns a small summary dict and
    persists the CQM + raw output to disk."""
    print(f"[{name}] formulating ...", flush=True)
    t0 = time.perf_counter()
    summary: dict = {"name": name, "ok": False, "elapsed_s": 0.0}
    try:
        provider = get_provider(name)
        if api_key is None and name != "local":
            summary["error"] = "no API key in env"
            return summary
        result = await provider.formulate(PROBLEM, api_key or "", timeout=timeout)
        elapsed = time.perf_counter() - t0
        summary["elapsed_s"] = elapsed
        summary["model"] = result.model
        summary["tokens"] = result.tokens_used
        summary["vars"] = len(result.cqm_json.get("variables", []))
        summary["constraints"] = len(result.cqm_json.get("constraints", []))
        summary["expected_optimum"] = (
            result.cqm_json.get("test_instance") or {}
        ).get("expected_optimum")

        cqm_path = OUT_DIR / f"cira_quantum_{name}_cqm.json"
        raw_path = OUT_DIR / f"cira_quantum_{name}_raw.txt"
        cqm_path.write_text(json.dumps(result.cqm_json, indent=2))
        raw_path.write_text(result.raw_llm_output)
        summary["cqm_path"] = str(cqm_path)
        summary["raw_path"] = str(raw_path)

        # Compile to confirm it's not just shape-valid but semantically OK.
        try:
            cqm, _registry, sense = compile_cqm_json(result.cqm_json)
            summary["compile"] = "OK"
            summary["compile_vars"] = len(cqm.variables)
            summary["compile_constraints"] = len(cqm.constraints)
            summary["sense"] = sense
        except Exception as e:
            summary["compile"] = f"FAIL: {str(e)[:120]}"

        summary["ok"] = summary.get("compile") == "OK"
        print(
            f"[{name}] done in {elapsed:.1f}s  "
            f"vars={summary['vars']}  constraints={summary['constraints']}  "
            f"expected={summary['expected_optimum']!r}  compile={summary.get('compile')}",
            flush=True,
        )
    except FormulationError as e:
        summary["elapsed_s"] = time.perf_counter() - t0
        summary["error"] = str(e)
        print(f"[{name}] FAILED: {e}", flush=True)
    except Exception as e:
        summary["elapsed_s"] = time.perf_counter() - t0
        summary["error"] = f"{type(e).__name__}: {e}"
        print(f"[{name}] UNEXPECTED: {summary['error']}", flush=True)
    return summary


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("claude", os.environ.get("ANTHROPIC_API_KEY"), 120),
        ("local",  None,                                 1200),
    ]

    results: list[dict] = []
    for name, key, timeout in targets:
        results.append(await _try_one(name, key, timeout))

    print()
    print(f"{'provider':<10} {'model':<28} {'time(s)':>8} {'vars':>5} "
          f"{'cnstr':>6} {'expected':>8} {'compile':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r and r.get("compile") is None:
            print(f"{r['name']:<10} {'-':<28} {r['elapsed_s']:>8.1f} "
                  f"{'-':>5} {'-':>6} {'-':>8} {'ERR':>8}  {r.get('error','')[:50]}")
        else:
            print(f"{r['name']:<10} {r.get('model','?'):<28} "
                  f"{r['elapsed_s']:>8.1f} {r.get('vars','?'):>5} "
                  f"{r.get('constraints','?'):>6} {r.get('expected_optimum','?')!s:>8} "
                  f"{r.get('compile','?'):>8}")
    print()
    for r in results:
        if r.get("ok"):
            print(f"  saved {r['name']}: {r['cqm_path']}")
            print(f"        {r['name']}: {r['raw_path']}")

    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
