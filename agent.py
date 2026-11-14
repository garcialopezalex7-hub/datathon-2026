"""
Agent híbrid ML + MPC (Programació Lineal a horitzó estès 48h) per al Repte UPC Sostenible.
Combina:
  1. Predicció LightGBM de demanda neta (load - pv) amb gestió de salts temporals.
  2. Predicció LightGBM de preu PVPC i emissions CO2 per a les hores t+24..t+47.
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
    """Agent híbrid LightGBM + MPC LP a horitzó estès (48 h)."""

    def __init__(self):
        self.eta = 0.9 ** 0.5
        self.cap = 2000.0
        self.p_max = 500.0
        # Pesos calibrats al ràtio oficial 70% CO2 / 30% Cost
        self.w_co2 = 0.9747
        self.w_price = 0.2526
        self.term_quantile = 0.55
        self.net_margin_kw = 50.0

        bundle_path = Path(__file__).parent / "model_bundle.pkl"
        bundle = joblib.load(bundle_path)
        self.model_net = bundle["model_net"]
        self.model_price = bundle["model_price"]
        self.model_co2 = bundle["model_co2"]
        for m in (self.model_net, self.model_price, self.model_co2):
            m.set_params(n_jobs=1)

        self.last_ts = None
        self.net_hist = deque(maxlen=24)
        self._mats_cache = {}

    def _get_matrices(self, horizon: int):
        """Construeix i guarda en memòria cau les matrius del problema LP."""
        if horizon in self._mats_cache:
            return self._mats_cache[horizon]

        # Variables: [c_0..c_{H-1}, d_0..d_{H-1}, g_0..g_{H-1}, soc_0..soc_{H-1}]
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
        tel = obs["noves_dades_5min"]
        last_net = float(tel["ultim_consum_kw"]) - float(tel["ultima_solar_kw"])

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

        if horizon <= 1:
            consigna = -self.p_max if soc_kwh > 0.0 else 0.0
        else:
            # 1. Estendre horitzó a 48h quan tenim finestra completa de 24h
            if horizon == 24:
                p_mean, p_std = float(prices.mean()), float(prices.std())
                c_mean, c_std = float(co2s.mean()), float(co2s.std())
                x_pc = np.empty((24, 11), dtype=float)
                for j in range(24):
                    ts_f = ts + timedelta(hours=24 + j)
                    dow = ts_f.weekday()
                    x_pc[j, 0] = j
                    x_pc[j, 1] = ts_f.month
                    x_pc[j, 2] = ts_f.hour
                    x_pc[j, 3] = dow
                    x_pc[j, 4] = 1.0 if dow >= 5 else 0.0
                    x_pc[j, 5] = prices[j]
                    x_pc[j, 6] = p_mean
                    x_pc[j, 7] = p_std
                    x_pc[j, 8] = co2s[j]
                    x_pc[j, 9] = c_mean
                    x_pc[j, 10] = c_std

                p_fut = np.clip(self.model_price.booster_.predict(x_pc), 0.0, None)
                c_fut = np.clip(self.model_co2.booster_.predict(x_pc), 0.0, None)
                p_all = np.concatenate([prices, p_fut])
                c_all = np.concatenate([co2s, c_fut])
            else:
                p_all = prices[:horizon]
                c_all = co2s[:horizon]

            h_tot = len(p_all)

            # 2. Predir demanda neta (load - pv) per a tot l'horitzó
            x_net = np.empty((h_tot, 9), dtype=float)
            for k in range(h_tot):
                ts_k = ts + timedelta(hours=k)
                dow = ts_k.weekday()
                x_net[k, 0] = ts_k.month
                x_net[k, 1] = ts_k.hour
                x_net[k, 2] = dow
                x_net[k, 3] = 1.0 if dow >= 5 else 0.0
                x_net[k, 4] = min(k, 23)
                x_net[k, 5] = p_all[k]
                x_net[k, 6] = c_all[k]
                x_net[k, 7] = last_net
                x_net[k, 8] = mean_24

            net_pred = np.maximum(0.0, self.model_net.booster_.predict(x_net) + self.net_margin_kw)

            # 3. Resoldre LP (MPC)
            w = self.w_co2 * c_all + self.w_price * p_all
            v_term = float(np.quantile(w, self.term_quantile)) * self.eta

            c_obj = np.zeros(4 * h_tot)
            c_obj[0:h_tot] = 1e-6
            c_obj[h_tot : 2 * h_tot] = 1e-6
            c_obj[2 * h_tot : 3 * h_tot] = w
            c_obj[4 * h_tot - 1] = -v_term

            a_eq, a_ub, bounds = self._get_matrices(h_tot)
            b_eq = np.zeros(h_tot)
            b_eq[0] = soc_kwh
            b_ub = -net_pred

            res = linprog(c_obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
            consigna = float(res.x[0] - res.x[h_tot]) if res.success else 0.0

        # 4. Clamping de seguretat física (0 infraccions garantides)
        eps = 1e-4
        max_carrega = max(0.0, min(self.p_max, (self.cap - soc_kwh) / self.eta - eps))
        max_descarrega = max(0.0, min(self.p_max, soc_kwh * self.eta - eps))
        return float(np.clip(consigna, -max_descarrega, max_carrega))
