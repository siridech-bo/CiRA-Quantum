"""Progress logging for long-running QRC jobs (durable + live HTML trace).

A multi-hour reproduction run needs three things a bare ``print`` doesn't
give: (1) durability — if the process dies you keep the events so far;
(2) step-level progress + ETA, not just per-order; (3) a way to *watch*
it without tailing a terminal.

:class:`ProgressLogger` provides all three with no server and no extra
deps: every event is appended to a JSONL file, echoed to stdout, and
re-rendered into a **self-contained HTML** page that ``<meta refresh>``es
itself — open it once in a browser (``file://``) and it live-updates as
the run writes new snapshots.

Usage::

    log = ProgressLogger(run_dir="artifacts/paper4_run", title="Paper-4 NARMA")
    log.event("start", "reproduction started", config={...})
    log.status(phase="NARMA10", step=120, total=600, eta_s=5400)
    log.event("order_done", "NARMA10 finished", nmse=3.4e-4)
    log.close("done")
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime
from pathlib import Path


class ProgressLogger:
    def __init__(self, run_dir: str, title: str, refresh_s: int = 10) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.refresh_s = refresh_s
        self.jsonl = self.dir / "events.jsonl"
        self.html = self.dir / "progress.html"
        self.jsonl.write_text("", encoding="utf-8")   # truncate
        self._events: list[dict] = []
        self._status: dict = {}
        self._t0 = time.time()
        self._results: dict[str, dict] = {}           # order/label -> metrics
        self.event("start", f"{title} started")

    # -- public API -----------------------------------------------------

    def event(self, kind: str, message: str, **data) -> None:
        """Record one event: append JSONL, print, re-render HTML."""
        ev = {
            "t": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - self._t0, 1),
            "kind": kind,
            "message": message,
            **data,
        }
        self._events.append(ev)
        with self.jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, default=float) + "\n")
        print(f"[{ev['elapsed_s']:>8.1f}s] {kind}: {message}", flush=True)
        self._render()

    def status(self, **fields) -> None:
        """Update the live status panel (phase/step/total/eta) — no console
        spam; only re-renders the HTML."""
        self._status.update(fields)
        self._render()

    def result(self, label: str, metrics: dict) -> None:
        """Record a finished sub-result (e.g. a NARMA order) for the table."""
        self._results[str(label)] = metrics
        self._render()

    def close(self, kind: str = "done", message: str = "") -> None:
        self.event(kind, message or f"{self.title} finished")

    # -- rendering ------------------------------------------------------

    def _render(self) -> None:
        self.html.write_text(self._html(), encoding="utf-8")

    def _html(self) -> str:
        st = self._status
        done, total = st.get("step"), st.get("total")
        pct = f"{100 * done / total:.1f}%" if done and total else "—"
        eta = st.get("eta_s")
        eta_str = _hms(eta) if eta else "—"
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td>"
            + "".join(f"<td>{_fmt(v)}</td>" for v in m.values())
            + "</tr>"
            for k, m in self._results.items()
        )
        headers = ""
        if self._results:
            first = next(iter(self._results.values()))
            headers = "<th>label</th>" + "".join(
                f"<th>{html.escape(str(h))}</th>" for h in first
            )
        log_rows = "".join(
            f"<tr><td class='t'>{e['elapsed_s']:.1f}s</td>"
            f"<td class='k'>{html.escape(e['kind'])}</td>"
            f"<td>{html.escape(e['message'])}</td></tr>"
            for e in reversed(self._events[-120:])
        )
        return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{self.refresh_s}">
<title>{html.escape(self.title)}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
  background:#0f1216;color:#e6e9ee}}
 .wrap{{max-width:960px;margin:0 auto;padding:28px 20px}}
 h1{{font-size:1.3rem;margin:.2em 0}} .muted{{color:#9aa4b2}}
 .panel{{background:#151a21;border:1px solid #2a3038;border-radius:10px;
  padding:14px 18px;margin:14px 0}}
 .big{{font-size:1.6rem;color:#6ea8fe;font-variant-numeric:tabular-nums}}
 table{{border-collapse:collapse;width:100%;font-size:.86rem;
  display:block;overflow-x:auto}}
 th,td{{border:1px solid #2a3038;padding:5px 9px;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#1b222c}}
 .log td{{text-align:left}} .log .t{{color:#9aa4b2;white-space:nowrap}}
 .log .k{{color:#6ea8fe}}
 .bar{{height:10px;background:#2a3038;border-radius:6px;overflow:hidden;margin-top:8px}}
 .bar>i{{display:block;height:100%;background:#2563eb;width:{pct if done and total else '0%'}}}
</style></head><body><div class="wrap">
<h1>{html.escape(self.title)}</h1>
<p class="muted">auto-refresh {self.refresh_s}s · elapsed {_hms(time.time()-self._t0)}
 · last update {datetime.now().strftime('%H:%M:%S')}</p>
<div class="panel">
 <div class="muted">phase: <b>{html.escape(str(st.get('phase','—')))}</b></div>
 <div class="big">{pct}</div>
 <div class="bar"><i></i></div>
 <div class="muted">step {done or '—'} / {total or '—'} · ETA {eta_str}</div>
</div>
{"<div class='panel'><b>Results</b><table><tr>" + headers + "</tr>" + rows + "</table></div>" if rows else ""}
<div class="panel"><b>Event log</b> (latest first)
 <table class="log">{log_rows}</table></div>
</div></body></html>"""


def _hms(sec) -> str:
    if sec is None:
        return "—"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3e}" if (v and abs(v) < 1e-2) else f"{v:.4f}"
    return html.escape(str(v))
