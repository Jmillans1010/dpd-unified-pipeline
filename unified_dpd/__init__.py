"""Integration without changing the original project APIs."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for project in ("framework_dpd", "procesador_iq_pa"):
    sys.path.insert(0, str(ROOT / project))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
__version__ = "0.3.0"
