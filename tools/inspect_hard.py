"""Inspecciona el comportament de l'agent en una setmana (Mode Difícil)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardmode_env import HardModeEnv, load_hourly_hard

agent_path = Path(sys.argv[1])
start = sys.argv[2]
days = int(sys.argv[3]) if len(sys.argv) > 3 else 7

spec = importlib.util.spec_from_file_location("cand", agent_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

df = load_hourly_hard()
block = df.loc[start:].iloc[: days * 24]

agent = mod.BatteryAgent()
eta = 0.9 ** 0.5
env = HardModeEnv(block)
obs = env.reset()
done = False
rows = []
while not done:
    a = agent.act(obs)
    t = env.inner.t
    row = block.iloc[t]
    soc_before = env.inner.soc
    rows.append({
        "ts": str(row.name)[:16],
        "hour": int(row.name.hour),
        "load": round(float(row["load_kw"]), 0),
        "pv": round(float(row["pv_kw"]), 0),
        "net": round(float(row["load_kw"] - row["pv_kw"]), 0),
        "price": round(float(row["precio_eur_per_kwh"]), 4),
        "co2": round(float(row["co2_kg_per_kwh"]), 4),
        "action": round(float(a), 1),
        "soc": round(float(soc_before), 0),
    })
    obs, done = env.step(a)

print("Setmana:", start, "->", days, "dies")
print("hora  load   pv    net  price   co2  action   soc")
for r in rows[:72]:
    print(f"{r['hour']:>3d}  {r['load']:>5.0f} {r['pv']:>5.0f} {r['net']:>6.0f} {r['price']:>6.4f} {r['co2']:>6.4f} {r['action']:>6.1f} {r['soc']:>5.0f}")

print("\nMètriques:")
for k, v in env.evaluate().items():
    print(f"  {k:22s}: {v}")
