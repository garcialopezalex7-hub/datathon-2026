"""
Agent MPC Estocastic de 2 Etapes + Incertesa Heteroscedastica + Calibracio per Horitzont (Mode Dificil).
"""

from collections import deque
from pathlib import Path
from typing import Any, Mapping
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, lil_matrix


class BatteryAgent:
    """Agente MPC estocastico de 2 etapas con incertidumbre heterocedastica solar."""

    def __init__(self) -> None:
        b = joblib.load(Path(__file__).resolve().parent / "model_bundle.pkl")
        self.m_l0 = lgb.Booster(model_str=b["m_l0_str"])
        self.m_l0_r = self._prep_ridge(b["m_l0_r"])
        self.m_pv0 = lgb.Booster(model_str=b["m_pv0_str"])
        self.m_net0 = lgb.Booster(model_str=b["m_net0_str"])
        self.m_net0_r = self._prep_ridge(b["m_net0_r"])
        self.q_res = {k: (np.asarray(d, dtype=float), np.asarray(n, dtype=float)) for k, (d, n) in b["q_res"].items()}
        self.alpha_k = np.asarray(b["alpha_k"], dtype=float)

        self.blocks_lp = b["blocks_lp"]
        self.lf_lgb = [lgb.Booster(model_str=s) for s in b["lf_lgb"]]
        self.lf_r = [self._prep_ridge(d) for d in b["lf_r"]]
        self.pvf_lgb = [lgb.Booster(model_str=s) for s in b["pvf_lgb"]]

        self.blocks_pc = b["blocks_pc"]
        self.p_lgb = [lgb.Booster(model_str=s) for s in b["p_lgb"]]
        self.p_r = [self._prep_ridge(d) for d in b["p_r"]]
        self.c_lgb = [lgb.Booster(model_str=s) for s in b["c_lgb"]]
        self.c_r = [self._prep_ridge(d) for d in b["c_r"]]

        self.l_hist: deque[float] = deque(maxlen=168)
        self.pv_hist: deque[float] = deque(maxlen=168)
        self.net_hist: deque[float] = deque(maxlen=168)
        self.p_hist: deque[float] = deque(maxlen=168)
        self.c_hist: deque[float] = deque(maxlen=168)
        self.last_ts: pd.Timestamp | None = None
        self._sum_co2_b = 0.0
        self._sum_cost_b = 0.0
        self._n_b = 0
        self._lp_cache: dict = {}

    @staticmethod
    def _prep_ridge(d: dict) -> tuple:
        return (
            np.asarray(d["mean"], dtype=float),
            np.asarray(d["scale"], dtype=float),
            np.asarray(d["coef"], dtype=float),
            float(d["intercept"]),
        )

    @staticmethod
    def _pred_ridge(rd: tuple, X: np.ndarray) -> np.ndarray:
        mean, scale, coef, intercept = rd
        return ((X - mean) / scale) @ coef + intercept

    @staticmethod
    def _time_features(ts: pd.Timestamp, ks: np.ndarray) -> tuple:
        hrs = (ts.hour + ks) % 24
        dows = (ts.dayofweek + (ts.hour + ks) // 24) % 7
        wks = (dows >= 5).astype(float)
        ang = (2.0 * np.pi / 24.0) * hrs
        return hrs, dows, wks, np.sin(ang), np.cos(ang), np.sin(2.0 * ang), np.cos(2.0 * ang)

    @staticmethod
    def _sanitize_series(raw: Any, fallback: float) -> np.ndarray:
        arr = np.asarray(raw, dtype=float)
        if arr.size == 0:
            return arr
        bad = ~np.isfinite(arr)
        if not np.any(bad):
            return arr
        out = arr.copy()
        last_ok = fallback
        for i in range(len(out)):
            if np.isfinite(out[i]):
                last_ok = float(out[i])
            else:
                out[i] = last_ok
        return out

    @staticmethod
    def _sf(val: Any, default: float) -> float:
        try:
            f = float(val)
            return f if np.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _lag_vec(arr: np.ndarray, ks: np.ndarray, offset: int, fallback: float) -> np.ndarray:
        n = len(arr)
        return np.array([arr[i] if 0 <= i < n else fallback for i in (n + ks - offset)], dtype=float)

    def _predict_net(self, h: int, ts: pd.Timestamp, l_last: float, pv_last: float,
                     irr: float, tamb: float, tpan: float, vent: float) -> tuple[np.ndarray, np.ndarray]:
        l_arr = np.asarray(self.l_hist, dtype=float)
        pv_arr = np.asarray(self.pv_hist, dtype=float)
        net_arr = np.asarray(self.net_hist, dtype=float)
        l_prev2 = l_arr[-2] if len(l_arr) >= 2 else l_last
        l_prev3 = l_arr[-3] if len(l_arr) >= 3 else l_prev2
        pv_prev2 = pv_arr[-2] if len(pv_arr) >= 2 else pv_last
        net_last = l_last - pv_last
        net_prev2 = net_arr[-2] if len(net_arr) >= 2 else net_last

        l_m6, l_m24, l_std24 = float(l_arr[-6:].mean()), float(l_arr[-24:].mean()), float(l_arr[-24:].std())
        l_m168 = float(l_arr.mean())
        pv_m24, pv_max24 = float(pv_arr[-24:].mean()), float(pv_arr[-24:].max())
        net_m6, net_m24, net_std24 = float(net_arr[-6:].mean()), float(net_arr[-24:].mean()), float(net_arr[-24:].std())

        ks = np.arange(h, dtype=int)
        hrs, dows, wks, sin_h, cos_h, sin_2h, cos_2h = self._time_features(ts, ks)

        l_lag24 = self._lag_vec(l_arr, ks, 24, l_m24)
        l_lag24_p = self._lag_vec(l_arr, ks, 25, l_m24)
        l_lag24_n = self._lag_vec(l_arr, ks, 23, l_m24)
        l_lag48 = self._lag_vec(l_arr, ks, 48, l_m24)
        l_lag168 = self._lag_vec(l_arr, ks, 168, l_m24)
        pv_lag24 = self._lag_vec(pv_arr, ks, 24, pv_m24)
        pv_lag24_p = self._lag_vec(pv_arr, ks, 25, pv_m24)
        pv_lag24_n = self._lag_vec(pv_arr, ks, 23, pv_m24)
        pv_lag48 = self._lag_vec(pv_arr, ks, 48, pv_m24)
        pv_lag168 = self._lag_vec(pv_arr, ks, 168, pv_m24)
        net_lag24 = self._lag_vec(net_arr, ks, 24, net_m24)
        net_lag168 = self._lag_vec(net_arr, ks, 168, net_m24)

        n_l = len(l_arr)
        l_yp = l_arr[-25] if n_l >= 25 else l_m24
        l_wp = l_arr[-169] if n_l >= 169 else l_yp
        pv_yp = pv_arr[-25] if n_l >= 25 else pv_m24

        X = np.column_stack([
            ks.astype(float), np.full(h, float(ts.month)), hrs.astype(float), dows.astype(float), wks,
            sin_h, cos_h, sin_2h, cos_2h, np.full(h, float(ts.hour)),
            np.full(h, l_last), np.full(h, l_last - l_prev2), np.full(h, l_prev2 - l_prev3),
            np.full(h, l_m6), np.full(h, l_m24), np.full(h, l_std24), np.full(h, l_m168),
            l_lag24, l_lag24_p, l_lag24_n, l_lag48, l_lag168, l_lag24 - l_yp, l_lag168 - l_wp,
            np.full(h, l_last - l_yp), np.full(h, l_last - l_wp), l_last + (l_lag24 - l_yp),
            np.full(h, pv_last), np.full(h, pv_last - pv_prev2), np.full(h, pv_m24), np.full(h, pv_max24),
            pv_lag24, pv_lag24_p, pv_lag24_n, pv_lag48, pv_lag168, pv_lag24 - pv_yp,
            np.full(h, pv_last / (pv_yp + 20.0)),
            np.full(h, net_last), np.full(h, net_last - net_prev2), np.full(h, net_m6), np.full(h, net_m24), np.full(h, net_std24),
            net_lag24, net_lag168, np.full(h, net_last - (l_yp - pv_yp)),
            np.full(h, irr), np.full(h, tamb), np.full(h, tpan), np.full(h, tpan - tamb), np.full(h, vent),
        ])

        net_pred = np.empty(h, dtype=float)
        pv_pred = np.empty(h, dtype=float)
        x0 = X[0:1]
        l0 = 0.85 * float(self.m_l0.predict(x0, num_threads=1)[0]) + 0.15 * float(self._pred_ridge(self.m_l0_r, x0)[0])
        pv0 = max(0.0, float(self.m_pv0.predict(x0, num_threads=1)[0]))
        net0_dir = 0.85 * float(self.m_net0.predict(x0, num_threads=1)[0]) + 0.15 * float(self._pred_ridge(self.m_net0_r, x0)[0])
        pv_pred[0] = pv0
        net_pred[0] = 0.5 * (l0 - pv0) + 0.5 * net0_dir

        for bi, (a, b) in enumerate(self.blocks_lp):
            if a >= h:
                break
            b_end = min(b, h)
            xb = X[a:b_end]
            pl = 0.85 * self.lf_lgb[bi].predict(xb, num_threads=1) + 0.15 * self._pred_ridge(self.lf_r[bi], xb)
            ppv = np.clip(self.pvf_lgb[bi].predict(xb, num_threads=1), 0.0, None)
            pv_pred[a:b_end] = ppv
            net_pred[a:b_end] = pl - ppv
        return net_pred, pv_pred

    def _predict_pc(self, h: int, ts: pd.Timestamp, p_known: np.ndarray, c_known: np.ndarray,
                    irr: float, tamb: float, vent: float) -> tuple[np.ndarray, np.ndarray]:
        p0, p1, p2 = float(p_known[0]), float(p_known[1]), float(p_known[2])
        c0, c1, c2 = float(c_known[0]), float(c_known[1]), float(c_known[2])
        p_arr = np.asarray(self.p_hist, dtype=float) if self.p_hist else np.array([p0])
        c_arr = np.asarray(self.c_hist, dtype=float) if self.c_hist else np.array([c0])

        p_m6, p_m24, p_std24 = float(p_arr[-6:].mean()), float(p_arr[-24:].mean()), float(p_arr[-24:].std())
        p_min24, p_max24, p_m168 = float(p_arr[-24:].min()), float(p_arr[-24:].max()), float(p_arr.mean())
        c_m6, c_m24, c_std24 = float(c_arr[-6:].mean()), float(c_arr[-24:].mean()), float(c_arr[-24:].std())
        c_min24, c_max24, c_m168 = float(c_arr[-24:].min()), float(c_arr[-24:].max()), float(c_arr.mean())

        n_p = len(p_arr)
        p_y2, c_y2 = (p_arr[-22] if n_p >= 22 else p_m24), (c_arr[-22] if n_p >= 22 else c_m24)
        p_y0, c_y0 = (p_arr[-24] if n_p >= 24 else p_m24), (c_arr[-24] if n_p >= 24 else c_m24)
        p_pos = (p2 - p_min24) / (p_max24 - p_min24 + 1e-6)
        c_pos = (c2 - c_min24) / (c_max24 - c_min24 + 1e-6)

        ks = np.arange(3, h, dtype=int)
        m_k = len(ks)
        hrs, dows, wks, sin_h, cos_h, sin_2h, cos_2h = self._time_features(ts, ks)

        p_lag24 = self._lag_vec(p_arr, ks, 24, p_m24)
        p_lag24_p = self._lag_vec(p_arr, ks, 25, p_m24)
        p_lag24_n = self._lag_vec(p_arr, ks, 23, p_m24)
        p_lag48 = self._lag_vec(p_arr, ks, 48, p_m24)
        p_lag168 = self._lag_vec(p_arr, ks, 168, p_m24)
        p_lag168_p = self._lag_vec(p_arr, ks, 169, p_m24)
        p_lag168_n = self._lag_vec(p_arr, ks, 167, p_m24)

        c_lag24 = self._lag_vec(c_arr, ks, 24, c_m24)
        c_lag24_p = self._lag_vec(c_arr, ks, 25, c_m24)
        c_lag24_n = self._lag_vec(c_arr, ks, 23, c_m24)
        c_lag48 = self._lag_vec(c_arr, ks, 48, c_m24)
        c_lag168 = self._lag_vec(c_arr, ks, 168, c_m24)
        c_lag168_p = self._lag_vec(c_arr, ks, 169, c_m24)
        c_lag168_n = self._lag_vec(c_arr, ks, 167, c_m24)

        X = np.column_stack([
            ks.astype(float), np.full(m_k, float(ts.month)), hrs.astype(float), dows.astype(float), wks,
            sin_h, cos_h, sin_2h, cos_2h,
            np.full(m_k, p0), np.full(m_k, p1), np.full(m_k, p2), np.full(m_k, p2 - p0), np.full(m_k, p2 - p1),
            np.full(m_k, p2 - p_y2), np.full(m_k, p0 - p_y0), np.full(m_k, p_m6), np.full(m_k, p_m24),
            np.full(m_k, p_std24), np.full(m_k, p_min24), np.full(m_k, p_max24), np.full(m_k, p_pos), np.full(m_k, p_m168),
            p_lag24, p_lag24_p, p_lag24_n, p_lag48, p_lag168, p_lag168_p, p_lag168_n,
            p2 + (p_lag24 - p_y2), (p_lag24 - p_m24) / (p_std24 + 1e-4),
            np.full(m_k, c0), np.full(m_k, c1), np.full(m_k, c2), np.full(m_k, c2 - c0), np.full(m_k, c2 - c1),
            np.full(m_k, c2 - c_y2), np.full(m_k, c0 - c_y0), np.full(m_k, c_m6), np.full(m_k, c_m24),
            np.full(m_k, c_std24), np.full(m_k, c_min24), np.full(m_k, c_max24), np.full(m_k, c_pos), np.full(m_k, c_m168),
            c_lag24, c_lag24_p, c_lag24_n, c_lag48, c_lag168, c_lag168_p, c_lag168_n,
            c2 + (c_lag24 - c_y2), (c_lag24 - c_m24) / (c_std24 + 1e-4),
            np.full(m_k, c0 / (p0 + 0.02)), np.full(m_k, c_m24 / (p_m24 + 0.02)),
            np.full(m_k, irr), np.full(m_k, tamb), np.full(m_k, vent),
        ])

        p_fut = np.empty(m_k, dtype=float)
        c_fut = np.empty(m_k, dtype=float)
        for bi, (a, b) in enumerate(self.blocks_pc):
            s_i, e_i = a - 3, min(b, h) - 3
            if s_i >= m_k:
                break
            xb = X[s_i:e_i]
            p_b = self.p_lgb[bi].predict(xb, num_threads=1, validate_features=False)
            c_b = self.c_lgb[bi].predict(xb, num_threads=1, validate_features=False)
            p_fut[s_i:e_i] = np.clip(0.85 * p_b + 0.15 * self._pred_ridge(self.p_r[bi], xb), 0.0, None)
            c_fut[s_i:e_i] = np.clip(0.75 * c_b + 0.25 * self._pred_ridge(self.c_r[bi], xb), 0.0, None)
        return np.concatenate([p_known[:3], p_fut]), np.concatenate([c_known[:3], c_fut])

    def _solve_stochastic_mpc(self, h: int, soc: float, cap: float, p_max: float, eta: float,
                              net_pred: np.ndarray, pv_pred: np.ndarray, w: np.ndarray, vt: float, hr0: int) -> float:
        sk = min(4, h)
        n_g = [5 if (k == 0 and sk >= 1) else (3 if k < sk else 1) for k in range(h)]
        tot_g = sum(n_g)
        key = (h, tuple(n_g), p_max, cap)
        if key not in self._lp_cache:
            nv = 2 * h + tot_g + h + 2
            ae, au = lil_matrix((h + 1, nv)), lil_matrix((tot_g, nv))
            soc_off = 2 * h + tot_g
            for k in range(h):
                ae[k, soc_off + k] = 1.0
                if k > 0:
                    ae[k, soc_off + k - 1] = -1.0
                ae[k, k], ae[k, h + k] = -eta, 1.0 / eta
            ae[h, soc_off + h - 1] = 1.0
            ae[h, soc_off + h] = -1.0
            ae[h, soc_off + h + 1] = -1.0
            ru, go = 0, 2 * h
            for k in range(h):
                for _ in range(n_g[k]):
                    au[ru, k], au[ru, h + k], au[ru, go] = 1.0, -1.0, -1.0
                    ru += 1
                    go += 1
            bnds = [(0.0, p_max)] * (2 * h) + [(0.0, None)] * tot_g + [(0.0, cap)] * h + [(0.0, cap * 0.5)] * 2
            self._lp_cache[key] = (csc_matrix(ae), csc_matrix(au), bnds, nv, soc_off)
        ae, au, bnds, nv, soc_off = self._lp_cache[key]

        pis5 = np.array([0.15, 0.20, 0.30, 0.20, 0.15])
        pis3 = np.array([0.25, 0.50, 0.25])
        c_obj = np.zeros(nv)
        c_obj[: 2 * h] = 1e-6
        bu = np.empty(tot_g)
        ru, go = 0, 2 * h
        for k in range(h):
            ng = n_g[k]
            if ng == 1:
                c_obj[go] = w[k]
                bu[ru] = -net_pred[k]
            else:
                pis = pis5 if ng == 5 else pis3
                q_sol, q_low = self.q_res[k]
                hr_k = (hr0 + k) % 24
                alpha_s = float(np.clip(pv_pred[k] / 500.0, 0.0, 1.0)) if (8 <= hr_k <= 18) else 0.0
                qs = (1.0 - alpha_s) * q_low + alpha_s * q_sol
                c_obj[go : go + ng] = pis * w[k]
                bu[ru : ru + ng] = -(net_pred[k] + qs)
            ru += ng
            go += ng
        c_obj[soc_off + h] = -1.15 * vt
        c_obj[soc_off + h + 1] = -0.80 * vt
        be = np.zeros(h + 1)
        be[0] = soc
        res = linprog(
            c_obj, A_ub=au, b_ub=bu, A_eq=ae, b_eq=be, bounds=bnds,
            method="highs", options={"presolve": False},
        )
        if not res.success or res.x is None:
            return 0.0
        val = float(res.x[0] - res.x[h])
        return val if np.isfinite(val) else 0.0

    def act(self, obs: Mapping[str, Any]) -> float:
        raw_soc = obs.get("soc_kwh")
        if raw_soc is None:
            raw_soc = obs.get("soc_actual_kwh")
        cap = self._sf(obs.get("cap_max_kwh"), 6000.0)
        soc = float(np.clip(self._sf(raw_soc, 0.0), 0.0, cap))
        p_max = self._sf(obs.get("power_max_kw"), 1500.0)
        eta = float(np.sqrt(0.90))

        raw_p = obs.get("preus_properes_24h")
        if raw_p is None:
            raw_p = obs.get("preus_horaris", [])
        raw_c = obs.get("co2_properes_24h")
        if raw_c is None:
            raw_c = obs.get("co2_horari", [])

        p_fallback = float(self.p_hist[-1]) if self.p_hist else 0.0
        c_fallback = float(self.c_hist[-1]) if self.c_hist else 0.0
        p_known = self._sanitize_series(raw_p, p_fallback)
        c_known = self._sanitize_series(raw_c, c_fallback)
        n_known = min(len(p_known), len(c_known))
        h = 24 if n_known >= 3 else max(1, n_known)

        raw_ts = obs.get("timestamp")
        try:
            ts = pd.Timestamp(raw_ts) if raw_ts is not None else pd.Timestamp("2024-01-01 00:00:00")
            if pd.isna(ts):
                ts = pd.Timestamp("2024-01-01 00:00:00")
        except (TypeError, ValueError):
            ts = pd.Timestamp("2024-01-01 00:00:00")

        if self.last_ts is not None:
            dt_sec = (ts - self.last_ts).total_seconds()
            if dt_sec < 0 or dt_sec > 7200:
                self.l_hist.clear(); self.pv_hist.clear(); self.net_hist.clear()
                self.p_hist.clear(); self.c_hist.clear()
                self._sum_co2_b = 0.0
                self._sum_cost_b = 0.0
                self._n_b = 0
        self.last_ts = ts

        tel = obs.get("noves_dades_5min") or obs
        l_last = self._sf(tel.get("ultim_consum_kw", obs.get("ultim_consum_kw")), 800.0)
        pv_last = self._sf(tel.get("ultima_solar_kw", obs.get("ultima_solar_kw")), 0.0)
        irr = self._sf(tel.get("irradiancia_wm2", obs.get("ultima_irradiancia_wm2")), 0.0)
        tamb = self._sf(tel.get("temp_ambient_c", obs.get("ultima_temp_ambient_c")), 18.0)
        tpan = self._sf(tel.get("temp_panel_c", obs.get("ultima_temp_panel_c")), 20.0)
        vent = self._sf(tel.get("vent_ms", obs.get("ultim_vent_ms")), 1.5)

        if self.p_hist and self.c_hist:
            bg = max(0.0, l_last - pv_last)
            self._sum_co2_b += bg * self.c_hist[-1]
            self._sum_cost_b += bg * self.p_hist[-1]
            self._n_b += 1

        self.l_hist.append(l_last)
        self.pv_hist.append(pv_last)
        self.net_hist.append(l_last - pv_last)

        if n_known <= 1:
            return -min(p_max, max(0.0, l_last - pv_last)) if soc > 0.0 else 0.0

        net_pred, pv_pred = self._predict_net(h, ts, l_last, pv_last, irr, tamb, tpan, vent)
        if n_known >= 24:
            p_win, c_win = p_known[:h], c_known[:h]
        elif h > 3 and n_known >= 3:
            p_win, c_win = self._predict_pc(h, ts, p_known, c_known, irr, tamb, vent)
        else:
            p_win, c_win = p_known[:h], c_known[:h]

        self.p_hist.append(float(p_known[0]))
        self.c_hist.append(float(c_known[0]))

        w_p = 0.3914
        prior_ratio = 1.44 / 0.3914
        if self._n_b >= 24 and self._sum_co2_b > 1e-6:
            emp_ratio = (0.7 / self._sum_co2_b) / (0.3 / self._sum_cost_b)
            blend = self._n_b / (self._n_b + 168.0)
            ratio = float(np.clip((1.0 - blend) * prior_ratio + blend * emp_ratio, 2.0, 8.0))
        else:
            ratio = prior_ratio
        w = (w_p * ratio) * c_win + w_p * p_win
        if h > 3:
            wm = float(w.mean())
            w = w.copy()
            w[3:] = np.clip(wm + self.alpha_k[3:h] * (w[3:] - wm), 0.0, None)

        vt = 0.0 if n_known < 3 else float(np.quantile(w, 0.55)) * eta
        cons = self._solve_stochastic_mpc(h, soc, cap, p_max, eta, net_pred, pv_pred, w, vt, ts.hour)

        eps = 1e-4
        max_chg = max(0.0, min(p_max, (cap - soc) / eta - eps))
        max_dis = max(0.0, min(p_max, soc * eta - eps))
        if not np.isfinite(cons):
            cons = 0.0
        return float(np.clip(cons, -max_dis, max_chg))
