"""Avalua un agent en Mode Difícil sobre data/train (o un subconjunt)."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardmode_env import load_hourly_hard, run_agent_hard

agent_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "agent_hard.py"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

spec = importlib.util.spec_from_file_location("candidate_agent", agent_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

df = load_hourly_hard()
if limit:
    df = df.iloc[:limit]

res = run_agent_hard(mod.BatteryAgent(), df)
print(f"Agent: {agent_path}  |  hores: {len(df)}")
for k, v in res.items():
    print(f"  {k:22s}: {v}")
