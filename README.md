# Datathon UPC 2026 — Optimización Inteligente de Batería (Campus Nord)

Solución para el reto **AED × UPC Sostenible (Datathon 2026)**: gestión horaria óptima de una batería industrial en el Campus Nord de la UPC para maximizar la descarbonización (**70% CO₂**) y el ahorro económico (**30% Coste eléctrico**) con **0 infracciones físicas**.

$$\text{Score} = 100 \cdot \left(0.7\,\Delta\text{CO}_2 + 0.3\,\Delta\text{Coste}\right) - 0.2 \cdot \text{Infracciones}$$

---

## Resumen Comparativo de Soluciones

### Modo Difícil (`6.000 kWh / ±1.500 kW`, η = 0.90, solo 3 h de precio/CO₂ + telemetría `t-1`)

| Implementación | Enfoque | Score validación | Infracciones | Tiempo (2.016 h) | Tamaño |
|---|---|---:|---:|---:|---:|
| **[`mode_dificil/submission.zip`](mode_dificil/submission.zip)** *(Recomendada)* | LightGBM + Ridge por bloques + MPC Estocástico + Pesos Adaptativos | **`18.94`** *(fuera de muestra, `93.1%` Oracle 24h `20.35`)* | **`0`** | **`18.04 s`** (`8.95 ms/paso`) | **`3.28 MB`** |
| [`submission_hard.zip`](submission_hard.zip) · [`agent_hard.py`](agent_hard.py) | Perfiles + k-NN analogía de días + MPC LP 24 h *(sin ML)* | `18.83` *(CV 7 días)* / `18.76` *(`train`)* | **`0`** | `~5 s` (`2.5 ms/paso`) | `36 KB` |

### Modo Estándar (`2.000 kWh / ±500 kW`, η = 0.90, 24 h de precio/CO₂ conocidas)

| Solución | Score (`train`) | Score Test oficial (2.016 h) | Infracciones |
|---|---:|---:|---:|
| Plantilla `starter_kit` | `0.00` | — | `46` |
| **[`mode_estandard/submission.zip`](mode_estandard/submission.zip)** (`MPC LP 24 h`) | **`5.64`** (`99.3%` Oracle `5.68`) | **`6.43`** (`-5.97%` CO₂ / `-7.50%` Coste) | **`0`** |

---

## Arquitectura de la Solución Principal (`mode_dificil/agent.py`)

```text
obs (t) ─► Sanitización & Memoria (deques 168h, detección de saltos de episodio)
              │
              ├──► Predicción Net Load & PV (0..23h): LightGBM + Ridge por bloques + k=0 híbrido
              ├──► Predicción Precio & CO₂ (3..23h): 3 bloques horarios (LightGBM + Ridge)
              │
              ▼
       Pesos Adaptativos Online (prior 70/30 + estimación empírica CO2_base / Cost_base)
       + Calibración por Horizonte (alpha_k) + Escenarios Heteroscedásticos (q_res según PV)
              │
              ▼
       MPC Estocástico LP (HiGHS, presolve=False, valor terminal cóncavo por tramos)
              │
              ▼
       Safety Clamp Físico (margen eps = 1e-4 kW respecto a ±1.500 kW y [0, 6.000] kWh) ─► action_kw
```

### Decisiones de Ingeniería Clave
1. **Modelado por bloques horarios (`blocks_pc = [(3, 7), (7, 13), (13, 24)]`):** Sustituye 42 llamadas secuenciales a LightGBM por 6 llamadas vectorizadas por bloque, reduciendo el tiempo total a `18.04 s` (2.016 pasos) y el tamaño del paquete a `3.28 MB`, con mejor generalización al entrenar cada bloque con 4–11× más muestras.
2. **Incertidumbre heteroscedástica en el MPC:** Los primeros pasos (`k = 0..3`) integran cuantiles de error residual (`q_res`) interpolados dinámicamente según la generación solar prevista (`pv_pred / 500`) en horas diurnas.
3. **Pesos adaptativos online:** Estima de forma incremental el ratio real `(0.7 / CO2_base) / (0.3 / Cost_base)` del episodio en curso mezclado con el prior calibrado (`1.44 / 0.3914`).
4. **Robustez de producción:** Gestión segura de `NaN`/`inf` mediante forward-fill, reinicio automático de buffers ante saltos temporales negativos o `> 2 h` entre semanas de evaluación, y validación estricta de finitud y cotas físicas.

### Alternativa Autocontenida sin ML (`agent_hard.py`)
* **Previsión de mercado y carga:** Perfiles mensuales/horarios + k-NN sobre 250 curvas diarias de `data/train/` para carga y CO₂, junto con índice de nubosidad persistente para FV.
* **MPC LP 24 h:** Restricción `grid_import >= net + c - d` con tablas incrustadas (`36 KB`, ~`5 s` para 2.016 pasos).

---

## Estructura del Repositorio

```text
.
├── mode_estandard/                      # Solución Modo Estándar (2.000 kWh, ±500 kW, 24 h)
│   ├── agent.py                         # Controlador MPC LP 24 h (HiGHS)
│   └── submission.zip                   # Paquete validado para Modo Estándar
├── mode_dificil/                        # Solución Principal Modo Difícil (6.000 kWh, ±1.500 kW, 3 h + meteo)
│   ├── agent.py                         # Agente MPC Estocástico + LightGBM/Ridge + Pesos Adaptativos
│   ├── model_bundle.pkl                 # Modelos serializados (3.35 MB)
│   └── submission.zip                   # Paquete validado para Modo Difícil (3.28 MB)
├── agent_hard.py                        # Alternativa autocontenida sin ML (k-NN analogía de días + LP)
├── hard_tables.pkl                      # Tablas auxiliares para agent_hard.py
├── submission_hard.zip                  # Paquete autocontenido (36 KB)
├── submission_mode_dificil.zip          # Paquete alternativo en raíz
├── tools/                               # Scripts de evaluación, CV, ablación y entrenamiento
├── paquet_participants_datathon_upc/    # Datos de entrenamiento, documentación y starter_kit oficial
├── requirements.txt                     # Dependencias Python del proyecto
├── AGENTS.md                            # Reglas de flujo de trabajo Git
└── README.md                            # Documentación técnica del proyecto
```

---

## Instalación, Evaluación y Validación Local

```powershell
# 1. Crear/activar entorno e instalar dependencias
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Validar paquetes de entrega con el verificador oficial
python paquet_participants_datathon_upc\starter_kit\check_submission.py mode_estandard\submission.zip
python paquet_participants_datathon_upc\starter_kit\check_submission.py mode_dificil\submission.zip
python paquet_participants_datathon_upc\starter_kit\check_submission.py submission_hard.zip

# 3. Evaluación y CV de la alternativa sin ML
python tools/eval_hard.py agent_hard.py
python tools/cv_hard.py agent_hard.py
```
