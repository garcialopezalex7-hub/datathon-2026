"""Empaqueta el Mode Difícil en submission_hard.zip (agent.py autocontingut)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "agent_hard.py"
OUT = ROOT / "submission_hard.zip"

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(SRC, "agent.py")

print(f"Creat {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
with zipfile.ZipFile(OUT) as z:
    print("Contingut:", [(n, z.getinfo(n).file_size) for n in z.namelist()])
