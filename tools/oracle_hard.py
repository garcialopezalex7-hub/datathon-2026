"""Compara agent vs MPC amb previsió perfecta vs oracle global (Mode Difícil)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardmode_env import HARD_SPEC, HardModeEnv, load_hourly_hard

ETA = 0.9 ** 0.5
CAP = HARD_SPEC["capacity_kwh"]
POWER = HARD_SPEC["power_kw"]
W_CO2 = 3.82
W_PRICE = 1.0


def solve_plan(net, w, soc0, cap=CAP, power=POWER, v_term=0.0):
    H = len(net)
    n = 4 * H
    a_eq = lil_matrix((H, n))
    a_ub = lil_matrix((H, n))
    for t in range(H):
        a_eq[t, 2 * H + t] = 1.0
        if t > 0:
            a_eq[t, 2 * H + t - 1] = -1.0
        a_eq[t, t] = -ETA
        a_eq[t, H + t] = 1.0 / ETA
        a_ub[t, 3 * H + t] = -1.0
        a_ub[t, t] = 1.0
        a_ub[t, H + t] = -1.0
    bounds = ([(0, power)] * H + [(0, power)] * H + [(0, cap)] * H + [(0, None)] * H)
    b_eq = np.zeros(H)
    b_eq[0] = soc0
    b_ub = -np.asarray(net, dtype=float)
    c = np.zeros(n)
    c[3 * H:] = w
    if v_term:
        c[3 * H - 1] = -v_term
    res = linprog(c, A_eq=csc_matrix(a_eq), b_eq=b_eq, A_ub=csc_matrix(a_ub), b_ub=b_ub,
                  bounds=bounds, method="highs")
    if not res.success:
        return np.zeros(H)
    return res.x[:H] - res.x[H:2 * H]


def simulate_actions(actions, block, soc0=3000.0):
    env = HardModeEnv(block)
    env.reset()
    env.inner.soc = soc0
    for a in actions:
        obs, done = env.step(float(a))
        if done:
            break
    return env.evaluate()


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-07-06"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    df = load_hourly_hard().loc[start:].iloc[: days * 24]
    net = (df["load_kw"] - df["pv_kw"]).values
    price = df["precio_eur_per_kwh"].values
    co2 = df["co2_kg_per_kwh"].values
    w = W_CO2 * co2 + W_PRICE * price

    print(f"Bloc {start} ({len(df)} h)")

    # 1) Oracle global
    actions = solve_plan(net, w, 3000.0)
    m = simulate_actions(actions, df)
    print("ORACLE        score=%6.2f  dCO2=%5.2f%%  dCost=%5.2f%%  infr=%d" %
          (m["score"], m["co2_estalviat_pct"], m["cost_estalviat_pct"], m["infraccions"]))

    # 2) MPC 24h amb previsió perfecta (valor terminal = quantil dels w futurs)
    acts, soc = [], 3000.0
    for t in range(len(df)):
        H = min(24, len(df) - t)
        v = float(np.quantile(w[t + H: t + H + 24], 0.6)) * ETA if t + H < len(w) else 0.0
        plan = solve_plan(net[t:t + H], w[t:t + H], soc, v_term=v)
        a = float(plan[0])
        acts.append(a)
        if a >= 0:
            soc = min(CAP, soc + a * ETA)
        else:
            soc = max(0.0, soc + a / ETA)
    m = simulate_actions(acts, df)
    print("PERFECT MPC   score=%6.2f  dCO2=%5.2f%%  dCost=%5.2f%%  infr=%d" %
          (m["score"], m["co2_estalviat_pct"], m["cost_estalviat_pct"], m["infraccions"]))

    # 3) Agent real
    spec = importlib.util.spec_from_file_location("cand", "agent_hard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = HardModeEnv(df)
    agent = mod.BatteryAgent()
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(agent.act(obs))
    m = env.evaluate()
    print("AGENT (3h)    score=%6.2f  dCO2=%5.2f%%  dCost=%5.2f%%  infr=%d" %
          (m["score"], m["co2_estalviat_pct"], m["cost_estalviat_pct"], m["infraccions"]))


if __name__ == "__main__":
    main()
