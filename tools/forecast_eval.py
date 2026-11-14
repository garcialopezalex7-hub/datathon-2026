"""Mesura l'error de les previsions de l'agent (preu, co2, load, pv, net) en Mode Difícil."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardmode_env import HardModeEnv, load_hourly_hard

spec = importlib.util.spec_from_file_location("cand", "agent_hard.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
agent = mod.BatteryAgent()

start = sys.argv[1] if len(sys.argv) > 1 else "2026-07-06"
days = int(sys.argv[2]) if len(sys.argv) > 2 else 7

df = load_hourly_hard().loc[start:].iloc[: days * 24]
env = HardModeEnv(df)
obs = env.reset()
done = False
errs = {"price": [], "co2": [], "load": [], "pv": [], "net": []}
i = 0
while not done and i < len(df):
    pf, cf, lf, vf = agent._forecast(obs, 24)
    actual = df.iloc[i: i + 24]
    n = len(actual)
    errs["price"].append((pf[:n] - actual["precio_eur_per_kwh"].values))
    errs["co2"].append((cf[:n] - actual["co2_kg_per_kwh"].values))
    errs["load"].append((lf[:n] - actual["load_kw"].values))
    errs["pv"].append((vf[:n] - actual["pv_kw"].values))
    errs["net"].append(((lf - vf)[:n] - (actual["load_kw"] - actual["pv_kw"]).values))
    obs, done = env.step(0.0)
    i += 1

print(f"Bloc {start} ({len(df)} h), previsions a 24 h")
for k, v in errs.items():
    e = np.concatenate(v)
    print(f"  {k:6s} MAE={np.mean(np.abs(e)):7.3f}  bias={np.mean(e):7.3f}  RMSE={np.sqrt(np.mean(e**2)):7.3f}  p90abs={np.quantile(np.abs(e),0.9):7.3f}")

# errors by horizon (1h, 3h, 12h, 24h)
print("\nMAE per horitzó:")
for k in (0, 2, 5, 11, 23):
    if k < 24:
        print(f"  k={k+1:2d}h  price={np.mean([np.abs(e[k]) for e in errs['price'] if len(e)>k]):.4f} "
              f"co2={np.mean([np.abs(e[k]) for e in errs['co2'] if len(e)>k]):.4f} "
              f"load={np.mean([np.abs(e[k]) for e in errs['load'] if len(e)>k]):.1f} "
              f"pv={np.mean([np.abs(e[k]) for e in errs['pv'] if len(e)>k]):.1f} "
              f"net={np.mean([np.abs(e[k]) for e in errs['net'] if len(e)>k]):.1f}")
