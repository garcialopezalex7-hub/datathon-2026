"""
Agent unificat MPC LP Estocàstic + LightGBM per a Mode Estàndard i Mode Difícil (Datathon UPC 2026).

Millores matemàtiques implementades per al Mode Difícil (6.000 kWh, ±1.500 kW, solar ×12, horitzó 3 h + meteo):
  1. Correcció d'asimetria de xarxa (Jensen / cost nul d'exportació):
     - Factor d'amplificació fotovoltaica (pv_mult = 1.10) i marge de seguretat sobre demanda neta
       (net_shift = -70.0 kW) per evitar descàrregues en buit cap a la xarxa (0 €/kWh) i maximitzar
       la captació d'excedents solars gratuïts.
  2. Fusió autorregressiva a l'hora actual k = 0 (p0_blend = 0.30):
     - Combina la predicció ML amb la telemetria immediata (ultim_consum_kw - ultima_solar_kw)
       per ancorar la restricció de descàrrega i autoconsum del pas actual.
  3. Des-suavitzat de variància (Variance Restoring, alpha_pc = 1.20) per a k = 3..23:
     - Restaura l'amplitud dels pics i valls de preu i CO2 en les hores extrapolades per LightGBM
       perquè el solver LP no subestimi l'arbitratge futur enfront de les 3 primeres hores exactes.
  4. Construcció vectoritzada de matrius de característiques NumPy i memòria cau de matrius disperses HiGHS.
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

        # Hiperparàmetres d'optimització sota incertesa (Mode Difícil)
        self.hard_pv_mult = 1.10
        self.hard_net_shift_kw = -70.0
        self.hard_p0_blend = 0.30
        self.hard_alpha_pc = 1.20

        bundle_path = Path(__file__).parent / "model_bundle.pkl"
        bundle = joblib.load(bundle_path)
        for model in bundle.values():
            model.set_params(n_jobs=1)

        self.std_net = bundle.get("std_net")
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
        alpha_pc: float = 1.0,
    ) -> float:
        w = w_co2 * c_win + w_price * p_win
        if alpha_pc != 1.0 and horizon > 3:
            w_mean = float(w.mean())
            w = w.copy()
            w[3:] = np.clip(w_mean + alpha_pc * (w[3:] - w_mean), 0.0, None)

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
        if self.std_net is not None:
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
        else:
            net_pred = np.full(horizon, max(p_max, last_net), dtype=float)

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

        # Predicció vectoritzada de demanda i solar per a les properes hores
        horizon_lp = 24 if rem >= 3 else max(1, rem)
        ks = np.arange(horizon_lp, dtype=float)
        hrs = (ts.hour + np.arange(horizon_lp)) % 24
        dows = (ts.day_of_week + (ts.hour + np.arange(horizon_lp)) // 24) % 7
        months = np.full(horizon_lp, ts.month, dtype=float)
        wknds = (dows >= 5).astype(float)
        sin_h = np.sin(2.0 * np.pi * hrs / 24.0)
        cos_h = np.cos(2.0 * np.pi * hrs / 24.0)

        l_lags = np.array(
            [self.load_hist[n_l + k - 24] if (n_l + k - 24) >= 0 else l_mean24 for k in range(horizon_lp)],
            dtype=float,
        )
        pv_lags = np.array(
            [self.pv_hist[n_l + k - 24] if (n_l + k - 24) >= 0 else pv_mean24 for k in range(horizon_lp)],
            dtype=float,
        )

        x_lp = np.column_stack(
            [
                ks,
                months,
                hrs,
                dows,
                wknds,
                sin_h,
                cos_h,
                np.full(horizon_lp, float(ts.hour)),
                np.full(horizon_lp, l_last),
                np.full(horizon_lp, l_mean24),
                l_lags,
                np.full(horizon_lp, pv_last),
                np.full(horizon_lp, pv_mean24),
                pv_lags,
                np.full(horizon_lp, irr_last),
                np.full(horizon_lp, tamb_last),
                np.full(horizon_lp, tpan_last),
                np.full(horizon_lp, vent_last),
            ]
        )

        load_pred = self.hard_load.booster_.predict(x_lp)
        pv_pred = np.clip(self.hard_pv.booster_.predict(x_lp), 0.0, None)

        # Correcció d'asimetria de xarxa i fusió autorregressiva a k = 0
        net_pred = load_pred - self.hard_pv_mult * pv_pred + self.hard_net_shift_kw
        if self.hard_p0_blend > 0.0:
            net_pred[0] = (1.0 - self.hard_p0_blend) * net_pred[0] + self.hard_p0_blend * (l_last - pv_last)

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

            p_lags = np.array(
                [self.price_hist[n_p + k - 24] if (n_p + k - 24) >= 0 else p_mean24 for k in range(3, 24)],
                dtype=float,
            )
            c_lags = np.array(
                [self.co2_hist[n_p + k - 24] if (n_p + k - 24) >= 0 else c_mean24 for k in range(3, 24)],
                dtype=float,
            )

            x_pc = np.column_stack(
                [
                    ks[3:],
                    months[3:],
                    hrs[3:],
                    dows[3:],
                    wknds[3:],
                    sin_h[3:],
                    cos_h[3:],
                    np.full(21, p0),
                    np.full(21, p1),
                    np.full(21, p2),
                    np.full(21, p2 - p0),
                    np.full(21, p_mean24),
                    np.full(21, p_min24),
                    np.full(21, p_max24),
                    p_lags,
                    np.full(21, c0),
                    np.full(21, c1),
                    np.full(21, c2),
                    np.full(21, c2 - c0),
                    np.full(21, c_mean24),
                    np.full(21, c_min24),
                    np.full(21, c_max24),
                    c_lags,
                ]
            )

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
            alpha_pc=self.hard_alpha_pc,
        )

    def act(self, obs: dict) -> float:
        """Determina la consigna de potència en kW per a l'hora actual."""
        ts = pd.Timestamp(obs["timestamp"])
        self._reset_if_gap(ts)

        tel = obs.get("noves_dades_5min") or {}
        is_hard = (
            self._safe_float(obs.get("cap_max_kwh"), 0.0) > 2500.0
            or "irradiancia_wm2" in tel
            or float(obs.get("soc_kwh", 0.0)) > 2000.0
        )

        default_cap = 6000.0 if is_hard else 2000.0
        default_pmax = 1500.0 if is_hard else 500.0
        cap_max = self._safe_float(obs.get("cap_max_kwh"), default_cap)
        p_max = self._safe_float(obs.get("power_max_kw"), default_pmax)
        soc_kwh = float(obs["soc_kwh"])

        if not is_hard and cap_max <= 2500.0:
            consigna = self._act_standard(obs, ts, cap_max, p_max)
        else:
            consigna = self._act_hard(obs, ts, cap_max, p_max)

        # Clamping de seguretat física (0 infraccions garantides)
        eps = 1e-4
        max_carrega = max(0.0, min(p_max, (cap_max - soc_kwh) / self.eta - eps))
        max_descarrega = max(0.0, min(p_max, soc_kwh * self.eta - eps))
        return float(np.clip(consigna, -max_descarrega, max_carrega))
