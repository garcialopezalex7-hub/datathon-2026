"""Cross-validació per blocs de 7 dies (composició semblant al test ocult: 12 setmanes/2.016 h)."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardmode_env import load_hourly_hard, run_agent_hard

DAYS = 7


def build_fold(df: pd.DataFrame, offset: int, days: int = DAYS) -> pd.DataFrame:
    blocks = []
    for month in sorted(set(df.index.to_period("M"))):
        start = pd.Timestamp(month.start_time) + pd.Timedelta(days=offset)
        end = start + pd.Timedelta(days=days) - pd.Timedelta(hours=1)
        block = df.loc[start:end]
        if len(block) >= 24:
            blocks.append(block)
    if not blocks:
        return df.iloc[:0]
    return pd.concat(blocks)


def load_agent(path):
    spec = importlib.util.spec_from_file_location("cand_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BatteryAgent


def evaluate_agent(agent_factory, df, offset=1):
    data = build_fold(df, offset=offset)
    return run_agent_hard(agent_factory(), data)


def evaluate_folds(agent_factory, df, offsets=(1, 8, 15), verbose=True):
    out = []
    for off in offsets:
        data = build_fold(df, off)
        res = run_agent_hard(agent_factory(), data)
        out.append(res)
        if verbose:
            print(f"  fold offset={off:2d}: hores={len(data):4d}  score={res['score']:6.2f} "
                  f"dCO2={res['co2_estalviat_pct']:5.2f}%  dCost={res['cost_estalviat_pct']:5.2f}%  "
                  f"infr={res['infraccions']}")
    return out


if __name__ == "__main__":
    agent_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "agent_hard.py"
    Agent = load_agent(agent_path)
    df = load_hourly_hard()
    rows = evaluate_folds(Agent, df)
    print("\nMitjana score: %.2f" % (sum(r["score"] for r in rows) / len(rows)))
