"""
Agent intel·ligent optimitzat per a l'optimització de la bateria del Campus Nord (Datathon UPC 2026).
Implementa un controlador MPC (Model Predictive Control) amb Programació Lineal (LP)
a horitzó mòbil de 24 hores sobre preus PVPC i emissions CO2, amb Safety Clamp físic.
Objectiu: minimitzar emissions CO2 (70%) i cost econòmic (30%) respecte al baseline sense bateria.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, lil_matrix


class BatteryAgent:
    """Agent MPC basat en Programació Lineal (HiGHS) amb tolerància zero a infraccions físiques."""

    def __init__(self):
        self.eta = 0.9 ** 0.5          # Rendiment carregada/descàrrega ≈ 0.9487
        self.cap = 2000.0              # Capacitat nominal kWh
        self.p_max = 500.0             # Potència màxima kW
        # Pesos alineats amb la puntuació oficial: 70% CO2, 30% Cost
        self.w_co2 = 0.7
        self.w_price = 0.3
        # Margen de seguretat (kW) per evitar infraccions límit per errors de coma flotant
        self.eps_power = 0.5
        self.eps_soc = 0.5             # Margen de SOC (kWh) per no operar als límits extremes
        # Escala del valor terminal
        self.v_term_scale = 1.0
        # Llindar mínim de diferència de cost per operar
        self.min_spread_charge = 0.008
        self.min_spread_discharge = 0.008
        # Número de hores més cares per al valor terminal
        self.top_n_hours = 3
        self._mats_cache = {}

    def _get_matrices(self, horizon: int):
        """Construeix i guarda en memòria cau les matrius d'igualtat per a un horitzó H."""
        if horizon in self._mats_cache:
            return self._mats_cache[horizon]

        n_vars = 3 * horizon
        a_eq = lil_matrix((horizon, n_vars))
        for t in range(horizon):
            # soc[t] - soc[t-1] - eta*charge[t] + (1/eta)*discharge[t] = b_eq[t]
            a_eq[t, 2 * horizon + t] = 1.0          # soc[t]
            if t > 0:
                a_eq[t, 2 * horizon + t - 1] = -1.0  # -soc[t-1]
            a_eq[t, t] = -self.eta                   # -eta * charge[t]
            a_eq[t, horizon + t] = 1.0 / self.eta     # (1/eta) * discharge[t]

        a_csc = csc_matrix(a_eq)
        bounds = (
            [(0.0, self.p_max)] * horizon      # charge[0:H] >= 0
            + [(0.0, self.p_max)] * horizon     # discharge[0:H] >= 0
            + [(0.0, self.cap)] * horizon       # soc[0:H] within [0, cap]
        )
        self._mats_cache[horizon] = (a_csc, bounds)
        return a_csc, bounds

    def act(self, obs: dict) -> float:
        """Determina la consigna de potència en kW per a l'hora actual.

        Retorna:
            > 0  -> Carregue la bateria des de la xarxa (kW positius)
            < 0  -> Descarrigue la bateria cap al campus (kW negatius)
            = 0  -> Mode esper
        """
        try:
            prices = np.asarray(obs["preus_properes_24h"], dtype=float)
            co2s = np.asarray(obs["co2_properes_24h"], dtype=float)
        except (KeyError, TypeError, ValueError):
            return 0.0

        # Filtrar valors no vàlids
        prices = np.nan_to_num(prices, nan=0.15, posinf=0.15, neginf=0.0)
        co2s = np.nan_to_num(co2s, nan=0.10, posinf=0.10, neginf=0.0)

        horizon = min(len(prices), len(co2s))
        soc_kwh = float(obs.get("soc_kwh", 1000.0))
        soc_kwh = max(0.0, min(self.cap, soc_kwh))

        # Cas degenerat: poc o cap horitzó
        if horizon <= 1:
            if soc_kwh > self.eps_soc:
                # Descàrrega conservadora si hi ha energia
                return -min(self.p_max, soc_kwh * self.eta - self.eps_soc)
            return 0.0

        # Cost combinat ponderat: 70% CO2 + 30% preu (alineat amb la puntuació oficial)
        w = self.w_co2 * co2s[:horizon] + self.w_price * prices[:horizon]

        # Valor terminal: valor de l'energia emmagatzemada al final de l'horitzó.
        # Utilitzem la mitjana de les N hores més cares (exclint l'hora actual) perquè
        # tenir bateria plena és més valuable abans de períodes caros/contaminants.
        future_costs = w[1:]
        if len(future_costs) > 0:
            top_n = min(self.top_n_hours, len(future_costs))
            v_term = float(np.mean(np.sort(future_costs)[-top_n:])) * self.eta * self.v_term_scale
        else:
            v_term = float(np.mean(w)) * self.eta

        # Funció objectiu (minimitzar):
        #   sum(w[t] * charge[t]) - sum(w[t] * discharge[t]) - v_term * soc[H-1]
        c_obj = np.zeros(3 * horizon)
        c_obj[0:horizon] = w                      # Cost de carregar a cada hora
        c_obj[horizon : 2 * horizon] = -w         # Benefici de descarregar (cost negatiu)
        c_obj[3 * horizon - 1] = -v_term          # Valor del SOC final (encouraja SOC alta abans de hores cares)

        a_eq, bounds = self._get_matrices(horizon)
        b_eq = np.zeros(horizon)
        b_eq[0] = soc_kwh

        try:
            res = linprog(
                c_obj, A_eq=a_eq, b_eq=b_eq,
                bounds=bounds, method="highs"
            )
            if res.success:
                consigna = float(res.x[0] - res.x[horizon])
            else:
                consigna = 0.0
        except Exception:
            consigna = 0.0

        # === Clamping de seguretat física (zero infraccions) ===
        # Límits per a una sola hora basats en SOC actual i rendiments
        # Espai per carregar: (cap - soc) / eta_c  (kWh / (kW·h) = kW)
        espai_carrega = max(0.0, self.cap - soc_kwh - self.eps_soc)
        max_carrega_kw = espai_carrega / self.eta
        max_carrega = max(0.0, min(self.p_max, max_carrega_kw - self.eps_power))

        # Espai per descarregar: soc * eta_d  (kWh * (kWh/kWh) = kWh -> kW per 1h)
        espai_descarrega = max(0.0, soc_kwh - self.eps_soc)
        max_descarrega_kw = espai_descarrega * self.eta
        max_descarrega = max(0.0, min(self.p_max, max_descarrega_kw - self.eps_power))

        consigna = float(np.clip(consigna, -max_descarrega, max_carrega))

        # === Gestió de SOC per franja horària ===
        # Evita descarregar massa abans de hores cares i deixa espai abans de hores barates
        hora = int(obs.get("hour", 0))

        # Hores de punta vesprina (18-22): no descarregar per sota del 20% SOC
        if 18 <= hora <= 22:
            min_soc_kwh = self.cap * 0.20
            if soc_kwh <= min_soc_kwh:
                max_descarrega = 0.0  # No descarregar més

        # Hores de càrrega barata (0-6 i 11-16): no carregar per sobre del 85% SOC
        if (0 <= hora <= 6) or (11 <= hora <= 16):
            max_soc_kwh = self.cap * 0.85
            if soc_kwh >= max_soc_kwh:
                max_carrega = 0.0  # No carregar més

        consigna = float(np.clip(consigna, -max_descarrega, max_carrega))

        # === Aplicar llindar de diferència mínima per evitar operacions marginals ===
        current_cost = w[0]
        max_future_cost = float(np.max(w[1:])) if horizon > 1 else current_cost
        min_future_cost = float(np.min(w[1:])) if horizon > 1 else current_cost

        # Només carregar si el cost actual és suficientment baix respecte al màxim futur
        if consigna > 0:
            spread = max_future_cost - current_cost
            if spread < self.min_spread_charge:
                consigna = 0.0  # No vale la pena carregar per diferència marginal

        # Només descarregar si el cost actual és suficientment alt respecte al mínim futur
        elif consigna < 0:
            spread = current_cost - min_future_cost
            if spread < self.min_spread_discharge:
                consigna = 0.0  # No vale la pena descarregar per diferència marginal

        # Consistencia final: evitar valors residuals per errors numèrics
        if abs(consigna) < 1e-6:
            consigna = 0.0

        return consigna
