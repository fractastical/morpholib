"""Regression gate for Problem-Fixer fix-008 (Zhang sorting math-KG Rule 3)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP_YAML = ROOT / "morphoeval/math_kg_loops/zhang-sorting-morphogenesis.yaml"


def test_zhang_sorting_loop_yaml_passes_verify_integrity() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "morphoeval/verify_integrity.py"),
            "--loop-yaml",
            str(LOOP_YAML),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
