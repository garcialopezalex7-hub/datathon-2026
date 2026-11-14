"""
Simulador local del "Mode Difícil" (Datathon UPC 2026 — Repte UPC Sostenible).

Reprodueix les regles publicades a https://repteupc.datathon.cat/ :
  - Fotovoltaica ampliada: pv_kw × 12
  - Bateria: 6.000 kWh, ±1.500 kW, SOC inicial 50%
  - obs["preus_properes_24h"] / obs["co2_properes_24h"] només amb 3 hores
  - obs["noves_dades_5min"] amb dades meteorològiques de l'última hora
  - obs inclou "cap_max_kwh" i "power_max_kw"
  - Excedent solar a 0 € i 0 kg CO2 (no remunerat)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "paquet_participants_datathon_upc" / "data" / "train"
KIT = ROOT / "paquet_participants_datathon_upc" / "starter_kit"

HARD_SPEC = {
    "capacity_kwh": 6000,
    "power_kw": 1500,
    "efficiency": 0.9,
    "soc_init": 0.5,
    "soc_min": 0.0,
    "soc_max": 1.0,
    "dt_hours": 1,
    "degradation": False,
}

METEO_COLS = {
    "irradiancia_wm2": "Atersa - IRRADIANCIA (W/m2)",
    "temp_ambient_c": "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)",
    "temp_panel_c": "Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)",
    "vent_ms": "Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)",
}


def _col(df, needle, extra_must=None):
    for c in df.columns:
        up = c.upper()
        if needle in up and (extra_must is None or all(m in up for m in extra_must)):
            return c
    raise KeyError(needle)


def load_hourly() -> pd.DataFrame:
    """Taula horària del campus (load, pv, preu, co2) + meteo horària (igual que el tutorial oficial)."""
    contador = pd.read_csv(DATA_DIR / "contador.csv", parse_dates=["Fecha"]).set_index("Fecha")
    inversor = pd.read_csv(DATA_DIR / "inversor.csv", parse_dates=["Fecha"]).set_index("Fecha")
    pvpc = pd.read_csv(DATA_DIR / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(DATA_DIR / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")
    meteo = pd.read_csv(DATA_DIR / "meteo.csv", parse_dates=["Fecha"]).set_index("Fecha")

    load = (-contador["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(lower=0, upper=5000)
    load_h = load.resample("1h").mean()

    pv_total = sum(inversor[_col(inversor, f"INVERSOR {n}", ["POTENCIA ACTIVA"])].fillna(0)
                   for n in (1, 2, 3)).clip(lower=0)
    pv_h = pv_total.resample("1h").mean()

    meteo_h = pd.DataFrame({k: meteo[v].resample("1h").mean() for k, v in METEO_COLS.items()})

    df = pd.DataFrame({
        "load_kw": load_h,
        "pv_kw": pv_h,
        "precio_eur_per_kwh": pvpc["precio_eur_per_kwh"],
        "co2_kg_per_kwh": co2["co2_kg_per_kwh"],
    }).join(meteo_h, how="left")

    # Físicament vàlid: neta les files sense mesures de load/pv
    df = df.dropna(subset=["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"])
    meteo_keys = list(METEO_COLS)
    df[meteo_keys] = df[meteo_keys].interpolate(limit_direction="both")
    return df


def load_hourly_hard(base: pd.DataFrame = None) -> pd.DataFrame:
    base = load_hourly() if base is None else base.copy()
    base["pv_kw"] = base["pv_kw"] * 12.0
    return base


class HardModeEnv:
    """Entorn seqüencial que genera obs en format 'Mode Difícil' sobre un CampusBatteryEnv oficial."""

    def __init__(self, hourly_data: pd.DataFrame, kit_env_class=None):
        if kit_env_class is None:
            import sys
            sys.path.insert(0, str(KIT))
            from environment import CampusBatteryEnv as kit_env_class
        self.inner = kit_env_class(hourly_data, spec=HARD_SPEC, export_price=0.0)
        self.weather = hourly_data[[c for c in METEO_COLS]] if set(METEO_COLS).issubset(hourly_data.columns) else None
        self.cap = float(HARD_SPEC["capacity_kwh"])
        self.power = float(HARD_SPEC["power_kw"])

    def reset(self):
        obs = self.inner.reset()
        return self._hard_obs(obs)

    def _hard_obs(self, obs):
        if obs is None:
            return None
        out = dict(obs)
        out["preus_properes_24h"] = list(obs["preus_properes_24h"])[:3]
        out["co2_properes_24h"] = list(obs["co2_properes_24h"])[:3]
        out["cap_max_kwh"] = self.cap
        out["power_max_kw"] = self.power

        noves = dict(obs["noves_dades_5min"])
        if self.weather is not None:
            t = int(self.inner.t)
            # Última hora mesurada: t-1 (o la mateixa hora si t=0)
            idx = max(t - 1, 0)
            for k in METEO_COLS:
                noves[k] = float(self.weather[k].iloc[idx]) if idx < len(self.weather) else 0.0
        out["noves_dades_5min"] = noves
        return out

    def step(self, action):
        obs, done = self.inner.step(action)
        return self._hard_obs(obs), done

    def evaluate(self, **kwargs):
        return self.inner.evaluate_results(**kwargs)


def run_agent_hard(agent, hourly_data: pd.DataFrame):
    env = HardModeEnv(hourly_data)
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.act(obs)
        noves = obs.get("noves_dades_5min", {})
        obs, done = env.step(action)
        steps += 1
    metrics = env.evaluate()
    metrics["steps"] = steps
    return metrics
