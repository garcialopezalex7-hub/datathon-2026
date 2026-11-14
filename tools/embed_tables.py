"""Incrusta les taules de previsió dins agent_hard.py (submissió en un sol fitxer)."""
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent_hard.py"
PKL = ROOT / "hard_tables.pkl"

with open(PKL, "rb") as f:
    T = pickle.load(f)


def mat(name, nd=4):
    a = np.round(np.asarray(T[name], dtype=float), nd)
    return ("np.array(" + np.array2string(a, separator=",", max_line_width=200, threshold=np.inf)
            + ", dtype=float)")


def vec(name, nd=4, dtype="float"):
    a = np.round(np.asarray(T[name], dtype=float), nd)
    if dtype == "int":
        a = np.asarray(a, dtype=int)
    return ("np.array(" + np.array2string(a, separator=",", max_line_width=200, threshold=np.inf)
            + f", dtype={dtype})")


lines = [
    "_EMBEDDED_TABLES = {",
    f"    'price_prof': {mat('price_prof', 5)},",
    f"    'co2_prof': {mat('co2_prof', 5)},",
    f"    'pv_prof': {mat('pv_prof', 1)},",
    f"    'irr_prof': {mat('irr_prof', 1)},",
    f"    'load_prof': {mat('load_prof', 1)},",
    f"    'load_dow': {vec('load_dow')},",
    f"    'price_dow': {vec('price_dow')},",
    f"    'co2_dow': {vec('co2_dow')},",
    f"    'ad_month': {vec('ad_month', dtype='int')},",
    f"    'ad_dow': {vec('ad_dow', dtype='int')},",
    f"    'ad_doy': {vec('ad_doy', dtype='int')},",
    f"    'ad_co2': {mat('ad_co2', 4)},",
    f"    'ad_load': {mat('ad_load', 4)},",
    "}",
]
block = "\n".join(lines)

src = AGENT.read_text(encoding="utf-8")
pattern = re.compile(r"_EMBEDDED_TABLES = \{.*?\n\}", re.DOTALL)
if not pattern.search(src):
    raise SystemExit("No s'ha trobat el bloc _EMBEDDED_TABLES")
src = pattern.sub(block, src, count=1)
AGENT.write_text(src, encoding="utf-8")
print(f"Incrustat a {AGENT} ({len(src)/1024:.1f} KB)")

