"""
Agent híbrid MPC LP 24 h + predictor LightGBM de demanda neta per al Mode Estàndard (Datathon UPC 2026).
Combina:
  1. Dades exactes a 24 h de preus PVPC i emissions CO2.
  2. Predictor LightGBM de demanda neta (load - pv) amb gestió de salts temporals.
  3. Optimitzador LP (HiGHS) calibrat al ràtio oficial 70% CO2 / 30% Cost.
  4. Safety Clamp físic per garantir 0 infraccions.
"""

from collections import deque
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, lil_matrix


class BatteryAgent:
    """Agent MPC LP 24 h + LightGBM demanda neta amb 0 infraccions (Mode Estàndard)."""

    def __init__(self):
        self.eta = 0.9 ** 0.5
        self.cap = 2000.0
        self.p_max = 500.0
        self.w_co2 = 0.9747
        self.w_price = 0.2526
        self.term_quantile = 0.55
        self.net_margin_kw = 50.0

        model_path = Path(__file__).parent / "model_net.pkl"
        self.model_net = joblib.load(model_path)
        self.model_net.set_params(n_jobs=1)

        self.last_ts = None
        self.net_hist = deque(maxlen=24)
        self._mats_cache = {}

    def _get_matrices(self, horizon: int):
        """Construeix i guarda en memòria cau les matrius del problema LP."""
        if horizon in self._mats_cache:
            return self._mats_cache[horizon]

        n_vars = 4 * horizon
        a_eq = lil_matrix((horizon, n_vars))
        a_ub = lil_matrix((horizon, n_vars))
        for k in range(horizon):
            a_eq[k, 3 * horizon + k] = 1.0
            if k > 0:
                a_eq[k, 3 * horizon + k - 1] = -1.0
            a_eq[k, k] = -self.eta
            a_eq[k, horizon + k] = 1.0 / self.eta

            a_ub[k, k] = 1.0
            a_ub[k, horizon + k] = -1.0
            a_ub[k, 2 * horizon + k] = -1.0

        bounds = (
            [(0.0, self.p_max)] * horizon
            + [(0.0, self.p_max)] * horizon
            + [(0.0, None)] * horizon
            + [(0.0, self.cap)] * horizon
        )
        mats = (csc_matrix(a_eq), csc_matrix(a_ub), bounds)
        self._mats_cache[horizon] = mats
        return mats

    def _update_history(self, obs: dict):
        """Actualitza l'històric de demanda neta reiniciant-lo si hi ha salt temporal."""
        ts = pd.Timestamp(obs["timestamp"])
        tel = obs.get("noves_dades_5min") or {}
        try:
            last_net = float(tel.get("ultim_consum_kw", self.p_max)) - float(tel.get("ultima_solar_kw", 0.0))
        except (TypeError, ValueError):
            last_net = self.p_max
        if not np.isfinite(last_net):
            last_net = self.p_max

        if self.last_ts is not None and (ts - self.last_ts) > timedelta(hours=1):
            self.net_hist.clear()

        self.net_hist.append(last_net)
        self.last_ts = ts
        mean_24 = float(sum(self.net_hist) / len(self.net_hist))
        return ts, last_net, mean_24

    def act(self, obs: dict) -> float:
        """Determina la consigna de potència en kW per a l'hora actual."""
        ts, last_net, mean_24 = self._update_history(obs)
        prices = np.asarray(obs["preus_properes_24h"], dtype=float)
        co2s = np.asarray(obs["co2_properes_24h"], dtype=float)
        horizon = min(len(prices), len(co2s))
        soc_kwh = float(obs["soc_kwh"])
        max_desc_net = float(np.clip(last_net, 0.0, self.p_max))

        if horizon <= 1:
            consigna = -max_desc_net if soc_kwh > 0.0 else 0.0
        else:
            p_win = prices[:horizon]
            c_win = co2s[:horizon]

            x_net = np.empty((horizon, 9), dtype=float)
            for k in range(horizon):
                ts_k = ts + timedelta(hours=k)
                dow = ts_k.weekday()
                x_net[k, 0] = ts_k.month
                x_net[k, 1] = ts_k.hour
                x_net[k, 2] = dow
                x_net[k, 3] = 1.0 if dow >= 5 else 0.0
                x_net[k, 4] = k
                x_net[k, 5] = p_win[k]
                x_net[k, 6] = c_win[k]
                x_net[k, 7] = last_net
                x_net[k, 8] = mean_24

            net_pred = self.model_net.booster_.predict(x_net) + self.net_margin_kw

            w = self.w_co2 * c_win + self.w_price * p_win
            ending = horizon < 24
            v_term = 0.0 if ending else float(np.quantile(w, self.term_quantile)) * self.eta

            c_obj = np.zeros(4 * horizon)
            c_obj[0:horizon] = 1e-6
            c_obj[horizon : 2 * horizon] = 1e-6
            c_obj[2 * horizon : 3 * horizon] = w
            c_obj[4 * horizon - 1] = -v_term

            a_eq, a_ub, bounds = self._get_matrices(horizon)
            bounds = list(bounds)
            bounds[horizon] = (0.0, max_desc_net)
            b_eq = np.zeros(horizon)
            b_eq[0] = soc_kwh
            b_ub = -net_pred

            res = linprog(c_obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.success:
                consigna = float(res.x[0] - res.x[horizon])
            else:
                med = float(np.median(w))
                consigna = self.p_max if w[0] < med else (-max_desc_net if w[0] > med else 0.0)

        eps = 1e-4
        max_carrega = max(0.0, min(self.p_max, (self.cap - soc_kwh) / self.eta - eps))
        max_descarrega = max(0.0, min(self.p_max, soc_kwh * self.eta - eps))
        return float(np.clip(consigna, -max_descarrega, max_carrega))
