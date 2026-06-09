#!/usr/bin/env python3
"""
Shared provenance helpers for wave-vector-analysis outputs.

Every major pipeline script should call ``record_run()`` after writing its
primary artifact(s). That stamps a sidecar ``<output>.provenance.json`` with:
  - generated_at (UTC + local)
  - script name + argv
  - git commit / branch / dirty flag
  - Python + key package versions
  - optional input file paths + mtimes

A rolling ``analysis_results/RUN_INDEX.jsonl`` log is also appended so the
full pipeline history is inspectable in one place.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "analysis_results"
RUN_INDEX = RESULTS / "RUN_INDEX.jsonl"

KEY_PACKAGES = (
    "numpy", "pandas", "matplotlib", "scipy", "tifffile", "openpyxl",
    "PyPDF2", "opencv-python", "Pillow",
)


def git_info(repo: Path = REPO) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    dirty = bool(run("status", "--porcelain"))
    return {
        "remote": run("remote", "get-url", "origin"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "commit_short": run("rev-parse", "--short", "HEAD"),
        "commit_message": run("log", "-1", "--format=%s"),
        "commit_date": run("log", "-1", "--format=%ci"),
        "dirty": dirty,
    }


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        import importlib.metadata as md
        for name in KEY_PACKAGES:
            try:
                out[name] = md.version(name)
            except Exception:
                pass
    except Exception:
        pass
    return out


def input_file_meta(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "bytes": st.st_size,
        "modified_local": dt.datetime.fromtimestamp(st.st_mtime).isoformat(
            timespec="seconds"),
    }


def collect(
    script: str,
    outputs: list[Path | str],
    *,
    inputs: dict[str, Path | str | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    return {
        "generated_at_utc": now.isoformat(timespec="seconds"),
        "generated_at_local": now.astimezone().isoformat(timespec="seconds"),
        "script": script,
        "argv": sys.argv,
        "repository": git_info(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "packages": package_versions(),
        "outputs": [str(o) for o in outputs],
        "inputs": {k: input_file_meta(v) for k, v in (inputs or {}).items()},
        "extra": extra or {},
    }


def sidecar_path(output: Path | str) -> Path:
    p = Path(output)
    return p.with_name(p.stem + ".provenance.json")


def write_sidecar(output: Path | str, record: dict[str, Any]) -> Path:
    path = sidecar_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def append_run_index(record: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"))
    with open(RUN_INDEX, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record_run(
    script: str,
    primary_output: Path | str,
    *,
    outputs: list[Path | str] | None = None,
    inputs: dict[str, Path | str | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write sidecar + append RUN_INDEX; return the provenance record."""
    outs = outputs or [primary_output]
    rec = collect(script, outs, inputs=inputs, extra=extra)
    sc = write_sidecar(primary_output, rec)
    append_run_index(rec)
    return rec


def summary_lines(record: dict[str, Any]) -> list[str]:
    git = record.get("repository", {})
    pkgs = record.get("packages", {})
    pkg_brief = ", ".join(f"{k}={v}" for k, v in sorted(pkgs.items())[:6])
    if len(pkgs) > 6:
        pkg_brief += ", …"
    dirty = " (DIRTY working tree)" if git.get("dirty") else ""
    return [
        f"Generated: {record.get('generated_at_utc', '?')} UTC",
        f"Script:    {record.get('script', '?')}",
        f"Git:       {git.get('commit_short', '?')} "
        f"{git.get('commit_message', '')}{dirty}",
        f"Python:    {record.get('python', {}).get('version', '?')}",
        f"Packages:  {pkg_brief or '(unknown)'}",
    ]


def csv_comment_header(record: dict[str, Any]) -> str:
    lines = ["# " + ln for ln in summary_lines(record)]
    lines.append("#")
    return "\n".join(lines) + "\n"
