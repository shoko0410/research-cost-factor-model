from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_python_module_qsf_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "qsf", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage: qsf" in result.stdout.lower()
