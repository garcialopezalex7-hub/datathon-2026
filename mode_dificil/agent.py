"""
Agent unificat MPC LP + LightGBM per a Mode Estàndard i Mode Difícil (Datathon UPC 2026).
Detecta automàticament la configuració de la bateria (cap_max_kwh / power_max_kw):
  - Mode Estàndard (2.000 kWh, ±500 kW, horitzó 24 h):
      Usa dades exactes de preu i CO2 a 24 h + predictor LightGBM de demanda neta.
  - Mode Difícil (6.000 kWh, ±1.500 kW, solar ×12, horitzó 3 h + meteo):
      Prediu demanda (load) i generació solar (pv) a 24 h amb telemetria meteo,
      extrapola preus i CO2 de +3 h a +23 h, i resol el problema LP òptim a 24 h.
En ambdós modes aplica Safety Clamp físic per garantir 0 infraccions.
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
    """Agent unificat compatible amb Mode Estàndard i Mode Difícil."""

    def __init__(self):
        self.eta = 0.9 ** 0.5
        self.term_quantile = 0.55
        self.std_net_margin_kw = 50.0

        bundle_path = Path(__file__).parent / "model_bundle.pkl"
        bundle = joblib.load(bundle_path)
        for model in bundle.values():
            model.set_params(n_jobs=1)

        self.std_net = bundle["std_net"]
        self.hard_load = bundle["hard_load"]
        self.hard_pv = bundle["hard_pv"]
        self.hard_price = bundle["hard_price"]
        self.hard_co2 = bundle["hard_co2"]

        self.last_ts = None
        self.net_hist = deque(maxlen=24)
        self.load_hist = deque(maxlen=24)
        self.pv_hist = deque(maxlen=24)
        self.price_hist = deque(maxlen=24)
        self.co2_hist = deque(maxlen=24)
        self._mats_cache = {}

    @staticmethod
    def _safe_float(val, default: float) -> float:
        try:
            f = float(val)
            return f if np.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    def _get_matrices(self, horizon: int, cap_max: float, p_max: float):
        """Construeix i guarda en memòria cau les matrius del problema LP."""
        key = (horizon, cap_max, p_max)
        if key in self._mats_cache:
            return self._mats_cache[key]

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
            [(0.0, p_max)] * horizon
            + [(0.0, p_max)] * horizon
            + [(0.0, None)] * horizon
            + [(0.0, cap_max)] * horizon
        )
        mats = (csc_matrix(a_eq), csc_matrix(a_ub), bounds)
        self._mats_cache[key] = mats
        return mats

    def _solve_lp(
        self,
        horizon: int,
        cap_max: float,
        p_max: float,
        soc_kwh: float,
        p_win: np.ndarray,
        c_win: np.ndarray,
        net_pred: np.ndarray,
        w_co2: float,
        w_price: float,
        ending: bool,
        max_desc_0: float | None = None,
    ) -> float:
        w = w_co2 * c_win + w_price * p_win
        v_term = 0.0 if ending else float(np.quantile(w, self.term_quantile)) * self.eta

        c_obj = np.zeros(4 * horizon)
        c_obj[0:horizon] = 1e-6
        c_obj[horizon : 2 * horizon] = 1e-6
        c_obj[2 * horizon : 3 * horizon] = w
        c_obj[4 * horizon - 1] = -v_term

        a_eq, a_ub, bounds = self._get_matrices(horizon, cap_max, p_max)
        if max_desc_0 is not None:
            bounds = list(bounds)
            bounds[horizon] = (0.0, max_desc_0)

        b_eq = np.zeros(horizon)
        b_eq[0] = soc_kwh
        b_ub = -net_pred

        res = linprog(c_obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success:
            return float(res.x[0] - res.x[horizon])

        med = float(np.median(w))
        fallback_d = max_desc_0 if max_desc_0 is not None else p_max
        return p_max if w[0] < med else (-fallback_d if w[0] > med else 0.0)

    def _reset_if_gap(self, ts: pd.Timestamp):
        if self.last_ts is not None and (ts - self.last_ts) > timedelta(hours=1):
            self.net_hist.clear()
            self.load_hist.clear()
            self.pv_hist.clear()
            self.price_hist.clear()
            self.co2_hist.clear()
        self.last_ts = ts

    def _act_standard(self, obs: dict, ts: pd.Timestamp, cap_max: float, p_max: float) -> float:
        tel = obs.get("noves_dades_5min") or {}
        last_net = self._safe_float(tel.get("ultim_consum_kw"), p_max) - self._safe_float(
            tel.get("ultima_solar_kw"), 0.0
        )
        self.net_hist.append(last_net)
        mean_24 = float(sum(self.net_hist) / len(self.net_hist))

        prices = np.asarray(obs["preus_properes_24h"], dtype=float)
        co2s = np.asarray(obs["co2_properes_24h"], dtype=float)
        horizon = min(len(prices), len(co2s))
        soc_kwh = float(obs["soc_kwh"])
        max_desc_net = float(np.clip(last_net, 0.0, p_max))

        if horizon <= 1:
            return -max_desc_net if soc_kwh > 0.0 else 0.0

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

        net_pred = self.std_net.booster_.predict(x_net) + self.std_net_margin_kw
        return self._solve_lp(
            horizon=horizon,
            cap_max=cap_max,
            p_max=p_max,
            soc_kwh=soc_kwh,
            p_win=p_win,
            c_win=c_win,
            net_pred=net_pred,
            w_co2=0.9747,
            w_price=0.2526,
            ending=horizon < 24,
            max_desc_0=max_desc_net,
        )

    def _act_hard(self, obs: dict, ts: pd.Timestamp, cap_max: float, p_max: float) -> float:
        tel = obs.get("noves_dades_5min") or {}
        l_last = self._safe_float(tel.get("ultim_consum_kw"), 800.0)
        pv_last = self._safe_float(tel.get("ultima_solar_kw"), 0.0)
        irr_last = self._safe_float(tel.get("irradiancia_wm2"), 0.0)
        tamb_last = self._safe_float(tel.get("temp_ambient_c"), 15.0)
        tpan_last = self._safe_float(tel.get("temp_panel_c"), 15.0)
        vent_last = self._safe_float(tel.get("vent_ms"), 2.0)

        prices = np.asarray(obs["preus_properes_24h"], dtype=float)
        co2s = np.asarray(obs["co2_properes_24h"], dtype=float)
        rem = min(len(prices), len(co2s))
        soc_kwh = float(obs["soc_kwh"])

        p0 = float(prices[0]) if rem >= 1 else 0.12
        c0 = float(co2s[0]) if rem >= 1 else 0.20

        # Estadístics de les darreres 24 h (abans d'afegir l'hora actual)
        if self.load_hist:
            l_mean24 = float(sum(self.load_hist) / len(self.load_hist))
            pv_mean24 = float(sum(self.pv_hist) / len(self.pv_hist))
        else:
            l_mean24, pv_mean24 = l_last, pv_last

        if self.price_hist:
            p_mean24 = float(sum(self.price_hist) / len(self.price_hist))
            p_min24 = float(min(self.price_hist))
            p_max24 = float(max(self.price_hist))
            c_mean24 = float(sum(self.co2_hist) / len(self.co2_hist))
            c_min24 = float(min(self.co2_hist))
            c_max24 = float(max(self.co2_hist))
        else:
            p_mean24 = p_min24 = p_max24 = p0
            c_mean24 = c_min24 = c_max24 = c0

        n_l = len(self.load_hist)
        n_p = len(self.price_hist)

        # Predicció de demanda i solar per a les properes hores
        horizon_lp = 24 if rem >= 3 else max(1, rem)
        x_lp = np.empty((horizon_lp, 18), dtype=float)
        ts0_hour = float(ts.hour)
        for k in range(horizon_lp):
            ts_k = ts + timedelta(hours=k)
            dow = ts_k.weekday()
            h_k = ts_k.hour
            sin_h = np.sin(2.0 * np.pi * h_k / 24.0)
            cos_h = np.cos(2.0 * np.pi * h_k / 24.0)

            idx_l = n_l + k - 24
            l_lag24 = self.load_hist[idx_l] if idx_l >= 0 else l_mean24
            pv_lag24 = self.pv_hist[idx_l] if idx_l >= 0 else pv_mean24

            x_lp[k, 0] = k
            x_lp[k, 1] = ts_k.month
            x_lp[k, 2] = h_k
            x_lp[k, 3] = dow
            x_lp[k, 4] = 1.0 if dow >= 5 else 0.0
            x_lp[k, 5] = sin_h
            x_lp[k, 6] = cos_h
            x_lp[k, 7] = ts0_hour
            x_lp[k, 8] = l_last
            x_lp[k, 9] = l_mean24
            x_lp[k, 10] = l_lag24
            x_lp[k, 11] = pv_last
            x_lp[k, 12] = pv_mean24
            x_lp[k, 13] = pv_lag24
            x_lp[k, 14] = irr_last
            x_lp[k, 15] = tamb_last
            x_lp[k, 16] = tpan_last
            x_lp[k, 17] = vent_last

        load_pred = self.hard_load.booster_.predict(x_lp)
        pv_pred = np.clip(self.hard_pv.booster_.predict(x_lp), 0.0, None)
        net_pred = load_pred - pv_pred

        # Actualitzem històrics per al següent pas
        self.load_hist.append(l_last)
        self.pv_hist.append(pv_last)
        self.price_hist.append(p0)
        self.co2_hist.append(c0)

        if rem <= 1:
            max_d0 = min(p_max, max(0.0, float(net_pred[0])))
            return -max_d0 if soc_kwh > 0.0 else 0.0

        if rem >= 24:
            p_win = prices[:24]
            c_win = co2s[:24]
            ending = False
        elif rem >= 3:
            p1, p2 = float(prices[1]), float(prices[2])
            c1, c2 = float(co2s[1]), float(co2s[2])
            p_diff = p2 - p0
            c_diff = c2 - c0

            x_pc = np.empty((21, 23), dtype=float)
            for idx_k, k in enumerate(range(3, 24)):
                ts_k = ts + timedelta(hours=k)
                dow = ts_k.weekday()
                h_k = ts_k.hour
                sin_h = np.sin(2.0 * np.pi * h_k / 24.0)
                cos_h = np.cos(2.0 * np.pi * h_k / 24.0)

                idx_p = n_p + k - 24
                p_lag24 = self.price_hist[idx_p] if idx_p >= 0 else p_mean24
                c_lag24 = self.co2_hist[idx_p] if idx_p >= 0 else c_mean24

                x_pc[idx_k, 0] = k
                x_pc[idx_k, 1] = ts_k.month
                x_pc[idx_k, 2] = h_k
                x_pc[idx_k, 3] = dow
                x_pc[idx_k, 4] = 1.0 if dow >= 5 else 0.0
                x_pc[idx_k, 5] = sin_h
                x_pc[idx_k, 6] = cos_h
                x_pc[idx_k, 7] = p0
                x_pc[idx_k, 8] = p1
                x_pc[idx_k, 9] = p2
                x_pc[idx_k, 10] = p_diff
                x_pc[idx_k, 11] = p_mean24
                x_pc[idx_k, 12] = p_min24
                x_pc[idx_k, 13] = p_max24
                x_pc[idx_k, 14] = p_lag24
                x_pc[idx_k, 15] = c0
                x_pc[idx_k, 16] = c1
                x_pc[idx_k, 17] = c2
                x_pc[idx_k, 18] = c_diff
                x_pc[idx_k, 19] = c_mean24
                x_pc[idx_k, 20] = c_min24
                x_pc[idx_k, 21] = c_max24
                x_pc[idx_k, 22] = c_lag24

            p_ext = np.clip(self.hard_price.booster_.predict(x_pc), 0.0, None)
            c_ext = np.clip(self.hard_co2.booster_.predict(x_pc), 0.0, None)
            p_win = np.concatenate([prices[:3], p_ext])
            c_win = np.concatenate([co2s[:3], c_ext])
            ending = False
        else:
            p_win = prices[:rem]
            c_win = co2s[:rem]
            ending = True

        horizon = min(len(p_win), len(c_win), len(net_pred))
        return self._solve_lp(
            horizon=horizon,
            cap_max=cap_max,
            p_max=p_max,
            soc_kwh=soc_kwh,
            p_win=p_win[:horizon],
            c_win=c_win[:horizon],
            net_pred=net_pred[:horizon],
            w_co2=1.4956,
            w_price=0.3914,
            ending=ending,
        )

    def act(self, obs: dict) -> float:
        """Determina la consigna de potència en kW per a l'hora actual."""
        ts = pd.Timestamp(obs["timestamp"])
        self._reset_if_gap(ts)

        cap_max = self._safe_float(obs.get("cap_max_kwh"), 2000.0)
        p_max = self._safe_float(obs.get("power_max_kw"), 500.0)
        soc_kwh = float(obs["soc_kwh"])

        if cap_max <= 2500.0:
            consigna = self._act_standard(obs, ts, cap_max, p_max)
        else:
            consigna = self._act_hard(obs, ts, cap_max, p_max)

        # Clamping de seguretat física (0 infraccions garantides)
        eps = 1e-4
        max_carrega = max(0.0, min(p_max, (cap_max - soc_kwh) / self.eta - eps))
        max_descarrega = max(0.0, min(p_max, soc_kwh * self.eta - eps))
        return float(np.clip(consigna, -max_descarrega, max_carrega))
