"""Genera les taules de previsió (perfil horari-mensual) per a l'agent del Mode Difícil."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardmode_env import load_hourly_hard

df = load_hourly_hard()
idx = df.index
month = idx.month - 1
hour = idx.hour
dow = idx.dayofweek

price = df["precio_eur_per_kwh"].values
co2 = df["co2_kg_per_kwh"].values
load = df["load_kw"].values
pv = df["pv_kw"].values
irr = df["irradiancia_wm2"].fillna(0).values

# Perfils robustos (mediana) per (mes, hora)
price_prof = np.zeros((12, 24))
co2_prof = np.zeros((12, 24))
pv_prof = np.zeros((12, 24))
irr_prof = np.zeros((12, 24))
load_prof = np.zeros((12, 24))
for m in range(12):
    for h in range(24):
        mask = (month == m) & (hour == h)
        if mask.sum() == 0:
            continue
        price_prof[m, h] = np.median(price[mask])
        co2_prof[m, h] = np.median(co2[mask])
        pv_prof[m, h] = np.median(pv[mask])
        irr_prof[m, h] = np.median(irr[mask])
        load_prof[m, h] = np.median(load[mask])

# Factor per dia de la setmana (sobre el perfil mes-hora)
load_dow = np.zeros(7)
for d in range(7):
    mask = dow == d
    expected = load_prof[month[mask], hour[mask]]
    load_dow[d] = np.median(load[mask] / np.maximum(expected, 1.0))
load_dow = load_dow / load_dow.mean()

# Factors de nivell del preu/co2 per dia de la setmana
price_dow = np.zeros(7)
co2_dow = np.zeros(7)
for d in range(7):
    mask = dow == d
    price_dow[d] = np.median(price[mask] / np.maximum(price_prof[month[mask], hour[mask]], 1e-6))
    co2_dow[d] = np.median(co2[mask] / np.maximum(co2_prof[month[mask], hour[mask]], 1e-9))
price_dow = price_dow / price_dow.mean()
co2_dow = co2_dow / co2_dow.mean()

tables = {
    "price_prof": price_prof,
    "co2_prof": co2_prof,
    "pv_prof": pv_prof,
    "irr_prof": irr_prof,
    "load_prof": load_prof,
    "load_dow": load_dow,
    "price_dow": price_dow,
    "co2_dow": co2_dow,
    "meta": {
        "n_hours": len(df),
        "mean_price": float(np.mean(price)),
        "mean_co2": float(np.mean(co2)),
        "mean_load": float(np.mean(load)),
        "mean_pv_hard": float(np.mean(pv)),
    },
}

# ---------------------------------------------------------------- analogies
# Llibreria de corbes diàries de CO2 normalitzades (per k-NN d'analogia de dies)
day_keys = sorted(set(df.index.normalize()))
ad_month, ad_dow, ad_doy, ad_co2, ad_load = [], [], [], [], []
for d in day_keys:
    sub = df.loc[d: d + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    if len(sub) < 20:
        continue
    m = d.month - 1
    dw = d.day_of_week
    curve = np.full(24, np.nan)
    curve_l = np.full(24, np.nan)
    for ts, row in sub.iterrows():
        h = ts.hour
        base_v = co2_prof[m, h] * co2_dow[dw]
        curve[h] = row["co2_kg_per_kwh"] / max(base_v, 1e-9)
        base_l = load_prof[m, h] * load_dow[dw]
        curve_l[h] = row["load_kw"] / max(base_l, 1.0)
    if np.sum(~np.isnan(curve)) < 20:
        continue
    ad_month.append(m)
    ad_dow.append(dw)
    ad_doy.append(int(d.dayofyear))
    ad_co2.append(np.nan_to_num(curve, nan=1.0))
    ad_load.append(np.nan_to_num(curve_l, nan=1.0))

tables["ad_month"] = np.array(ad_month, dtype=int)
tables["ad_dow"] = np.array(ad_dow, dtype=int)
tables["ad_doy"] = np.array(ad_doy, dtype=int)
tables["ad_co2"] = np.array(ad_co2, dtype=float)
tables["ad_load"] = np.array(ad_load, dtype=float)

with open("/tmp/hard_tables.pkl", "wb") as f:

    pickle.dump(tables, f)

print("Taules generades per a", len(df), "hores")
print("mean price %.4f  co2 %.4f  load %.1f  pv_hard %.1f" %
      (tables["meta"]["mean_price"], tables["meta"]["mean_co2"], tables["meta"]["mean_load"], tables["meta"]["mean_pv_hard"]))
print("load_dow:", np.round(load_dow, 3))
print("price_dow:", np.round(price_dow, 3))
print("co2_dow:", np.round(co2_dow, 3))
print("PV hard max per mes:", np.round(pv_prof.max(axis=1), 1))

# Denominadors del baseline en Mode Difícil (per calibrar pesos de l'objectiu)
net = np.maximum(load - pv, 0.0)
co2_base = float(np.sum(net * co2))
cost_base = float(np.sum(net * price))
print("\nBaseline hard (train): co2=%.1f kg  cost=%.1f EUR  hores=%d" % (co2_base, cost_base, len(df)))
print("Ratio 0.7/co2_base vs 0.3/cost_base -> w_co2/w_price =",
      (0.7 / co2_base) / (0.3 / cost_base))
