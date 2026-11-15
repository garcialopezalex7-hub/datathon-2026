# Guion de Exposición, Teoría y Q&A — Modo Difícil (Datathon UPC 2026)

Documento de apoyo para presentar [presentation_campus.html](presentation_campus.html) en **4 minutos** entre **2 personas** (la **Diapositiva 1** se deja en pantalla mientras os colocáis; el cronómetro arranca al pasar a la **Diapositiva 2**).

---

## 1. Reparto 50 / 50 (2 min cada uno)

| Tramo | Persona | Diapositivas | Rol en directo |
|---|---|---|---|
| **0:00 – 2:00** | **Persona A** | Diapositivas **2 y 3** | Explica el reto, los límites y cómo funciona la solución (arquitectura). |
| **2:00 – 4:00** | **Persona B** | Diapositivas **4, 5 y 6** | Explica los resultados verificados, las mejoras futuras y el cierre. |

* **Control de diapositivas:** Quien no está hablando pasa las diapositivas (y pulsa `⏱ 04:00` al empezar la Diapositiva 2).
* **Reparto de Q&A:**
  * **Persona A:** Preguntas técnicas de modelo, física, MPC, `NaN`/bordes y código.
  * **Persona B:** Preguntas de métricas, validación fuera de muestra, Oracle 24h, tiempos y mejoras futuras.

---

## 2. Guion Hablado Intuitivo (4 minutos)

### Persona A (`0:00 – 2:00`)

#### Diapositiva 2 · El Desafío (`0:00 – 0:50`)
> *"Gestionamos la batería del Campus Nord —6.000 kWh, como unas 100 baterías de coche eléctrico— con el objetivo de reducir un 70% el CO₂ y un 30% la factura eléctrica, cumpliendo tres límites duros: **0 infracciones**, menos de **25 MB** y menos de **60 segundos** para 12 semanas.*
>
> *¿Por qué es difícil el Modo Difícil? Porque es como **conducir de noche con unas luces que solo alumbran 3 horas por delante**: solo vemos las próximas 3 horas de precio y CO₂ y la última lectura de consumo, sol y tiempo. Las otras 21 horas están a oscuras.*
>
> *Además, de día las nubes hacen que la energía solar varíe muchísimo, y nuestro primer prototipo tardaba **37.2 segundos** en local: demasiado cerca del límite de 60 segundos."*

#### Diapositiva 3 · Nuestra Solución (`0:50 – 2:00`)
> *"Para resolverlo montamos un sistema en tres pasos que funciona como un **GPS inteligente**:*
> 1. ***Predecir las 21 horas que faltan (Forecasting por bloques):** En vez de usar 21 modelos pequeños y lentos para cada hora futura, los agrupamos en **3 bloques horarios** (mañana, tarde y noche) combinando LightGBM y Ridge. Pasamos de 42 llamadas por hora a solo 6: mucho más rápido y entrenando con hasta 11 veces más datos por modelo.*
> 2. ***Decidir con margen de seguridad (MPC Estocástico + Pesos Online):** En cada hora trazamos un plan óptimo de 24 horas con programación lineal (HiGHS), pero **solo aplicamos el primer paso** y a la hora siguiente recalculamos con la realidad, igual que un GPS. Además, de día añadimos escenarios de error más amplios por si pasan nubes, y ajustamos dinámicamente el peso entre CO₂ y coste según cómo venga la semana.*
> 3. ***Blindaje físico:** Antes de enviar la orden a la batería, un filtro final recorta cualquier exceso por mínimo que sea. Es matemáticamente imposible cometer una infracción.*
>
> *Ahora [Nombre de B] os enseña qué resultados consigue esta arquitectura y hasta dónde podría llegar."*

---

### Persona B (`2:00 – 4:00`)

#### Diapositiva 4 · Resultados (`2:00 – 2:55`)
> *"¿Qué conseguimos con esto?*
>
> *Evaluando fuera de muestra sobre 2.016 horas (12 semanas), logramos **18.94 puntos**, reduciendo casi un **19% tanto el CO₂ como el coste**, con **cero infracciones**.*
>
> *Para saber cómo de cerca estamos del límite físico, calculamos el **Oracle**: un sistema ideal que hiciera trampa y conociera el futuro exacto de las 24 horas. Ese techo teórico da 20.35. Es decir, con solo 3 horas de visibilidad, **alcanzamos el 93.1% del límite teórico perfecto**.*
>
> *Y en rendimiento: bajamos el tiempo a la mitad (**de 37.2 s a 18.0 s**) y el paquete de **10.76 MB a 3.28 MB**, con margen de sobra bajo los 60 segundos."*

#### Diapositiva 5 · Posibles Mejoras (`2:55 – 3:35`)
> *"¿Cómo podríamos rascar ese último 6.9% hasta el Oracle?*
> 1. ***Planificación en 2 etapas real:** Permitir que el optimizador adapte su respuesta a cada escenario futuro dentro del mismo problema lineal, en lugar de buscar una única trayectoria común.*
> 2. ***Prever la meteorología futura:** Estimar si habrá nubes o cambios de temperatura en las próximas horas en lugar de asumir que se mantiene la lectura de `t−1`.*
> 3. ***Reutilizar el optimizador en memoria:** Actualizar solo los precios nuevos en HiGHS sin reconstruir el LP cada hora, bajando el tiempo total de 18 segundos a menos de 10."*

#### Diapositiva 6 · Cierre (`3:35 – 4:00`)
> *"En resumen: un agente que alcanza el **93.1% del óptimo teórico**, en **18 segundos** y con **0 infracciones** físicas, robusto ante datos anómalos y listo para producción. Muchas gracias, quedamos abiertos a vuestras preguntas."*

---

## 3. Teoría Necesaria para Defender el Proyecto

### 3.1 Física de la batería y Programación Lineal (LP)
* **Eficiencia ida/vuelta ($\eta_{\text{rt}} = 0.90$):** Se reparte simétricamente entre carga y descarga como $\eta = \sqrt{0.90} \approx 0.9487$.
  * Al cargar $p_c$ kW, la energía almacenada aumenta en $\eta \cdot p_c$.
  * Al descargar $p_d$ kW, la energía almacenada disminuye en $p_d / \eta$.
* **Variables separadas ($p_c \ge 0,\; p_d \ge 0$):** Si usáramos una sola variable con signo, la eficiencia cambiaría según el signo y la ecuación dejaría de ser lineal. Separando carga y descarga (con una penalización mínima $10^{-6}$ para evitar carga y descarga simultánea), el problema es **100% lineal (LP)** y se resuelve exactamente en milisegundos con **HiGHS**.
* **Safety Clamp (`eps = 1e-4` kW):** Recorta la acción final respecto a `±1.500 kW` y respecto a `(cap - soc) / eta` y `soc * eta`, evitando infracciones por redondeo de coma flotante en `SoC = 0` o `SoC = 6000`.

### 3.2 Control Predictivo por Modelo (MPC) y Valor Terminal
* **Horizonte móvil (*Receding Horizon*):** En cada paso $t$, resolvemos la trayectoria óptima para las próximas 24 h ($k = 0 \dots 23$), pero **solo aplicamos $k = 0$**. En $t + 1$ incorporamos la telemetría real y re-optimizamos.
* **Valor terminal cóncavo (`vt`):** Sin valoración al final de la ventana ($k = 23$), el LP vaciaría la batería al 0% al final de cada horizonte. Asignamos un valor `vt = quantil_55(w) * eta` dividido en dos tramos (`1.15 * vt` para `[0, 50%]` de SoC y `0.80 * vt` para `[50%, 100%]`): mantiene reserva estratégica sin saturar la batería.

### 3.3 Incertidumbre Heteroscedástica (`q_res`)
* *Heteroscedástico* significa que la varianza del error **depende del estado**: de noche el error de carga neta es bajo ($\approx \pm 40\text{ kW}$), pero de día con generación fotovoltaica crece hasta $\approx \pm 210\text{ kW}$.
* En los primeros pasos ($k = 0 \dots 3$), el LP evalúa cuantiles de error residual interpolando entre el perfil nocturno y el solar según `pv_pred[k] / 500` entre las 8 h y las 18 h.

### 3.4 Modelos Híbridos por Bloques (`LightGBM + Ridge`)
* **Por qué híbrido:** `LightGBM` captura relaciones no lineales (hora, estacionalidad, PV) y `Ridge` estabiliza tendencias y extrapolaciones.
* **Por qué 3 bloques (`[(3, 7), (7, 13), (13, 24)]`):** Unificar horizontes pasando `k` como feature reduce de 42 a 6 llamadas a LightGBM por paso (bajando el tiempo total de `37.24 s` a `18.04 s` y el zip a `3.28 MB`) y multiplica por 4–11× las muestras de entrenamiento por modelo.

### 3.5 Pesos Adaptativos Online (`CO₂` vs `Coste`)
* El score oficial mide **ahorros relativos** respecto al baseline sin batería: `0.7 * d_CO2 + 0.3 * d_Cost`.
* La tasa marginal real entre 1 kg de CO₂ y 1 € depende del ratio `(0.7 / CO2_base) / (0.3 / Cost_base)` de cada episodio. El agente estima este ratio online con los datos observados, suavizado con el prior calibrado (`1.44 / 0.3914`) para evitar ruido en el arranque.

---

## 4. Posibles Preguntas del Jurado y Respuestas

1. **¿Por qué usáis MPC + LightGBM en vez de Reinforcement Learning (PPO / DQN)?**
   > *"Porque la dinámica de la batería y sus límites físicos son exactos y lineales. Un solver LP como HiGHS garantiza el óptimo matemático exacto del horizonte y **0 infracciones físicas**, algo difícil de asegurar con RL puro sin capas extra de corrección, además de ejecutarse en 18 segundos."*
2. **¿Cómo sabéis que `18.94` es un buen resultado?**
   > *"Porque medimos el **Oracle 24h** sobre las mismas 2.016 horas: un controlador ideal que conociera exactamente el futuro de consumo, solar, precios y CO₂ de las próximas 24 horas logra **20.35**. Con solo 3 horas de mercado y telemetría pasada, alcanzamos el **93.1% del límite teórico**."*
3. **¿Por qué agrupar en 3 bloques mejora la velocidad sin destruir la precisión?**
   > *"Al pasar el horizonte `k` como feature, cada bloque entrena con entre 4 y 11 veces más datos que un modelo aislado por hora, reduciendo el sobreajuste. Y pasar de 42 a 6 invocaciones de LightGBM por paso elimina la sobrecarga FFI de Python/C++."*
4. **¿Qué ocurre si llegan `NaN`, infinitos o si el evaluador cambia de semana?**
   > *"Los precios y CO₂ pasan por sanitización forward-fill, la salida del LP tiene guarda de finitud y el agente detecta saltos temporales negativos o mayores a 2 horas para reiniciar los historiales de 168 h entre bloques semanales."*
5. **¿Cómo funciona la calibración `alpha_k` por horizonte?**
   > *"Los modelos de ML tienden a suavizar hacia la media condicional en horizontes lejanos (`k >= 3`). `alpha_k` restaura la amplitud real de las variaciones futuras `(w[3:] - wm)` para que el MPC no infravalore los picos de precio/CO₂ de horas posteriores."*
