"""Build the Pepper voice references using the shared preparation workflow."""

import runpy
import sys
from pathlib import Path

folder = Path(__file__).resolve().parent
sys.argv = [str(folder.parent / "prepare-reference.py"), "--folder", str(folder), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
