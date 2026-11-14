"""
Controlador MPC Lineal (HiGHS) a 24h per a la bateria del Campus Nord (Datathon UPC 2026).
Calibrat a 70% CO2 / 30% Cost, amb buidatge final, límit de demanda neta i 0 infraccions.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, lil_matrix


class BatteryAgent:
    def __init__(self):
        self.eta = 0.9 ** 0.5
        self.cap, self.p_max = 2000.0, 500.0
        self.w_co2, self.w_price, self.q_term = 0.9747, 0.2526, 0.55
        self._cache = {}

    def _get_matrices(self, h: int):
        if h not in self._cache:
            a = lil_matrix((h, 3 * h))
            for t in range(h):
                a[t, 2 * h + t] = 1.0
                if t > 0:
                    a[t, 2 * h + t - 1] = -1.0
                a[t, t], a[t, h + t] = -self.eta, 1.0 / self.eta
            b = [(0.0, self.p_max)] * (2 * h) + [(0.0, self.cap)] * h
            self._cache[h] = (csc_matrix(a), b)
        return self._cache[h]

    def act(self, obs: dict) -> float:
        prices = np.asarray(obs["preus_properes_24h"], dtype=float)
        co2s = np.asarray(obs["co2_properes_24h"], dtype=float)
        h = min(len(prices), len(co2s))
        soc = float(obs["soc_kwh"])

        tel = obs["noves_dades_5min"]
        net_kw = float(np.clip(tel["ultim_consum_kw"] - tel["ultima_solar_kw"], 0.0, self.p_max))

        if h <= 1:
            p_kw = -net_kw if soc > 0.0 else 0.0
        else:
            w = self.w_co2 * co2s[:h] + self.w_price * prices[:h]
            v_term = 0.0 if h < 24 else float(np.quantile(w, self.q_term)) * self.eta

            c = np.concatenate([w + 1e-6, -w + 1e-6, np.zeros(h)])
            c[-1] = -v_term

            a_eq, bounds = self._get_matrices(h)
            bounds = list(bounds)
            bounds[h] = (0.0, net_kw)
            b_eq = np.zeros(h)
            b_eq[0] = soc

            res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
            p_kw = float(res.x[0] - res.x[h]) if res.success else 0.0

        eps = 1e-4
        max_c = max(0.0, min(self.p_max, (self.cap - soc) / self.eta - eps))
        max_d = max(0.0, min(self.p_max, soc * self.eta - eps))
        return float(np.clip(p_kw, -max_d, max_c))
