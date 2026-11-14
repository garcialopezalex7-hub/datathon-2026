"""Escombrat d'hiperparàmetres de l'agent de Mode Difícil sobre folds de 7 dies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_hard import evaluate_folds, load_agent
from hardmode_env import load_hourly_hard


def factory(Agent, **overrides):
    def make():
        a = Agent()
        for k, v in overrides.items():
            setattr(a, k, v)
        return a
    return make


def main():
    Agent = load_agent(Path(__file__).resolve().parent.parent / "agent_hard.py")
    df = load_hourly_hard()
    offsets = (1, 8, 15)

    grids = {
        "w_analog_load": [0.5, 0.75, 1.0],
        "w_analog_co2": [0.25, 0.5, 0.75],
        "margin_kw": [0.0, 30.0, 60.0, 90.0],
    }
    base = {}
    results = []
    for name, values in grids.items():
        for v in values:
            kw = dict(base)
            kw.update({name: v})
            print(f"\n=== {name}={v} ===")
            rows = evaluate_folds(factory(Agent, **kw), df, offsets=offsets, verbose=True)
            mean = sum(r["score"] for r in rows) / len(rows)
            results.append((mean, name, v))
            print(f"  -> mitjana {mean:.2f}")

    print("\n=== RESUM ===")
    for mean, name, v in sorted(results, reverse=True):
        print(f"{mean:7.2f}  {name}={v}")


if __name__ == "__main__":
    main()
