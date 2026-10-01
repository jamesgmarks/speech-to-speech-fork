"""Generate the Pepper demo using the shared single-call preview workflow."""

import runpy
import sys
from pathlib import Path

folder = Path(__file__).resolve().parent
sys.argv = [str(folder.parent / "generate-preview.py"), "--folder", str(folder), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
