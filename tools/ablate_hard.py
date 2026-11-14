"""Ablació: quant guanya l'agent si encerta (a posteriori) preus/CO2 o el net load."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardmode_env import HardModeEnv, load_hourly_hard

spec = importlib.util.spec_from_file_location("cand", "agent_hard.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DF = None


class Ablated(mod.BatteryAgent):
    actual_w = False
    actual_net = False

    def _forecast(self, obs, horizon):
        pf, cf, lf, vf = super()._forecast(obs, horizon)
        ts = obs.get("timestamp")
        if ts is None:
            return pf, cf, lf, vf
        try:
            idx = DF.index.get_loc(ts)
        except KeyError:
            return pf, cf, lf, vf
        n = min(horizon, len(DF) - idx)
        if self.actual_w:
            pf[:n] = DF["precio_eur_per_kwh"].values[idx: idx + n]
            cf[:n] = DF["co2_kg_per_kwh"].values[idx: idx + n]
        if self.actual_net:
            lf[:n] = DF["load_kw"].values[idx: idx + n]
            vf[:n] = DF["pv_kw"].values[idx: idx + n]
        return pf, cf, lf, vf


def run(cls, df):
    env = HardModeEnv(df)
    agent = cls()
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(agent.act(obs))
    return env.evaluate()


def main():
    global DF
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-07-06"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    DF = load_hourly_hard().loc[start:].iloc[: days * 24]
    for label, attrs in [
        ("real agent      ", {}),
        ("actual w        ", {"actual_w": True}),
        ("actual net      ", {"actual_net": True}),
        ("actual w+net    ", {"actual_w": True, "actual_net": True}),
    ]:
        cls = type("A", (Ablated,), attrs)
        m = run(cls, DF)
        print(f"{label} score={m['score']:6.2f}  dCO2={m['co2_estalviat_pct']:5.2f}%  "
              f"dCost={m['cost_estalviat_pct']:5.2f}%  infr={m['infraccions']}")


if __name__ == "__main__":
    main()
