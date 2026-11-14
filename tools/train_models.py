"""Entrena models ridge multi-horitzó (1..24 h) per a preu i CO2 (Mode Difícil)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardmode_env import load_hourly_hard

H = 24
LAM = 1.0

df = load_hourly_hard()
idx = df.index
price = df["precio_eur_per_kwh"].values
co2 = df["co2_kg_per_kwh"].values
wind = df["vent_ms"].fillna(0).values
irr = df["irradiancia_wm2"].fillna(0).values
temp = df["temp_ambient_c"].fillna(20).values

month = idx.month - 1
hour = idx.hour
dow = idx.day_of_week

# Perfils (mediana) per mes/hora i factors dow, igual que l'agent
price_prof = np.zeros((12, 24))
co2_prof = np.zeros((12, 24))
for m in range(12):
    for h in range(24):
        mask = (month == m) & (hour == h)
        price_prof[m, h] = np.median(price[mask]) if mask.sum() else 0.14
        co2_prof[m, h] = np.median(co2[mask]) if mask.sum() else 0.085

price_dow = np.array([np.median(price[dow == d] / np.maximum(price_prof[month[dow == d], hour[dow == d]], 1e-6)) for d in range(7)])
price_dow /= price_dow.mean()
co2_dow = np.array([np.median(co2[dow == d] / np.maximum(co2_prof[month[dow == d], hour[dow == d]], 1e-9)) for d in range(7)])
co2_dow /= co2_dow.mean()


def base_profile(prof, dowf, m, h, d):
    return prof[m, h] * dowf[d]


def make_samples():
    rows = []
    for t in range(len(df) - 1):
        # només si les següents 24 h són contigües
        if t + H >= len(df):
            break
        dt = (idx[t + 1] - idx[t]).total_seconds()
        if dt != 3600:
            continue
        if (idx[t + H] - idx[t]).total_seconds() != H * 3600:
            continue
        m0, h0, d0 = month[t], hour[t], dow[t]
        for target, prof, dowf, arr in (("price", price_prof, price_dow, price),
                                       ("co2", co2_prof, co2_dow, co2)):
            rows.append((target, t, m0, h0, d0, prof, dowf, arr))
    return rows


def features(t, k, m0, h0, d0, prof, dowf, arr):
    m_t = month[t + k]
    h_t = hour[t + k]
    d_t = dow[t + k]
    base_t0 = base_profile(prof, dowf, m0, h0, d0)
    base_t = base_profile(prof, dowf, m_t, h_t, d_t)
    vals = arr[t - 2: t + 1] if t >= 2 else np.array([arr[t]])
    base_past = np.array([base_profile(prof, dowf, month[t - j], hour[t - j], dow[t - j])
                          for j in range(min(3, t + 1))])
    ratios = np.log(np.maximum(vals[::-1], 1e-6) / np.maximum(base_past, 1e-6))
    r_current = ratios[0]
    r_mean = float(np.mean(ratios))
    x = np.array([
        1.0,
        r_current,
        r_mean,
        np.sin(2 * np.pi * h_t / 24), np.cos(2 * np.pi * h_t / 24),
        np.sin(2 * np.pi * d_t / 7), np.cos(2 * np.pi * d_t / 7),
        np.sin(2 * np.pi * m_t / 12), np.cos(2 * np.pi * m_t / 12),
        np.log(max(base_t, 1e-6)),
        wind[t] / 10.0,
        irr[t] / 800.0,
        (temp[t] - 20.0) / 10.0,
        k / 24.0,
    ])
    y = arr[t + k] / max(base_t, 1e-6)
    return x, y


def build_dataset(target):
    X = {k: [] for k in range(1, H + 1)}
    Y = {k: [] for k in range(1, H + 1)}
    for t in range(2, len(df) - H):
        if (idx[t + H] - idx[t]).total_seconds() != H * 3600:
            continue
        m0, h0, d0 = month[t], hour[t], dow[t]
        prof, dowf, arr = (price_prof, price_dow, price) if target == "price" else (co2_prof, co2_dow, co2)
        for k in range(1, H + 1):
            x, y = features(t, k, m0, h0, d0, prof, dowf, arr)
            X[k].append(x)
            Y[k].append(y)
    return {k: (np.array(v), np.array(Y[k])) for k, v in X.items()}


def ridge_fit(X, y, lam):
    XtX = X.T @ X
    XtX += lam * np.eye(XtX.shape[0])
    XtX[0, 0] -= lam  # no penalitzar intercept
    return np.linalg.solve(XtX, X.T @ y)


def evaluate(target):
    data = build_dataset(target)
    # validació: últims 25% de les mostres (temporal)
    coefs = {}
    maes = []
    for k in range(1, H + 1):
        X, y = data[k]
        n_train = int(len(X) * 0.75)
        beta = ridge_fit(X[:n_train], y[:n_train], LAM)
        pred = np.clip(X[n_train:] @ beta, 0.0, None)
        prof_val = np.exp(X[n_train:, 9])  # valor del perfil al moment objectiu
        mae_ratio = np.mean(np.abs(pred - y[n_train:]))
        # MAE en unitats reals aproximat: |pred-profile - y-profile| * profile
        err = np.abs((pred - y[n_train:]) * prof_val)
        maes.append(err.mean())
        coefs[k] = beta
    print(f"{target}: MAE temporal (últim 25%) = {np.mean(maes):.4f} sobre h=1..24")
    return coefs


for target in ("price", "co2"):
    coefs = evaluate(target)
    np.save(f"/tmp/ridge_{target}.npy", np.array([coefs[k] for k in range(1, H + 1)]))
    print(f"  coefs desats a /tmp/ridge_{target}.npy")
