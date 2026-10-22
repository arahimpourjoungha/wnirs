"""The README's quick start must keep running."""
import subprocess
import sys
from pathlib import Path


def test_quickstart_runs_and_finds_the_effect():
    script = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "contrast 'incongruent - congruent'" in out.stdout
