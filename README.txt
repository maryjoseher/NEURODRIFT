##################################################################################################################################
Paquete para adquisición con 2 NI independientes.
Conexión por defecto:
- ECG: Dev1/ai0
- EDA: Dev2/ai1

Cambios clave:
- physio_ni.py usa 2 tareas NI independientes, una por dispositivo.
- physio_runtime.py es RAW real: sin HRV, sin SCR, sin picos.
- app.py conserva consola, tracking ocular y bloques, y guarda el CSV crudo dentro de raw_physio/ al usar Guardar Sesion.

Si tus nombres en NI MAX no coinciden, edita en app.py la inicialización de self.physio_cfg.
##################################################################################################################################

# NEURODRIFT

Sistema multimodal para la estimación de deriva atencional en tiempo casi real mediante la integración de señales fisiológicas (ECG, EDA) y comportamiento ocular (FaceMesh).

---

## 1. Descripción

NeuroDrift es una plataforma de monitoreo cognitivo que estima el estado atencional de un usuario a partir de biomarcadores fisiológicos y visuales. 

El sistema no mide atención de forma directa, sino que **infiere un estado latente de deriva atencional** mediante:

- Activación autonómica (ECG + EDA)
- Inestabilidad ocular (PERCLOS + variabilidad del iris)
- Modelado probabilístico (Filtro de Kalman con estado aumentado)

El resultado final es un índice continuo:

> **I_k ∈ [0,1] → nivel de deriva atencional**

---

## 2. Arquitectura del sistema

El sistema está dividido en módulos independientes:

###  Adquisición
- ECG → NI USB (Dev1/ai0)
- EDA → NI USB (Dev2/ai1)
- Video → Webcam + MediaPipe FaceMesh

Cada señal se sincroniza mediante un **reloj maestro (time.time())**, permitiendo alineación multimodal real. :contentReference[oaicite:0]{index=0}

---

###  Procesamiento de señales

#### ECG
- Filtrado: bandpass 0.5–40 Hz + notch 50/60 Hz :contentReference[oaicite:1]{index=1}
- Detección de picos R
- Extracción:
  - HR
  - RMSSD (variabilidad cardíaca)

#### EDA
- Señal tónica (SCL)
- Conteo de eventos fásicos (SCR)

#### Ocular
- Landmarks de FaceMesh (478 puntos) :contentReference[oaicite:2]{index=2}
- Métricas:
  - EAR (Eye Aspect Ratio)
  - PERCLOS
  - Variabilidad del iris
- Validación:
  - Rechazo por pose
  - Interpolación de parpadeos
  - Filtro de saltos oculares

---

###  Ventanas temporales

Las características se calculan en ventanas deslizantes:

- Ventana: 60 s
- Hop: 30 s

Vector de características:

φ = [HR, ln(RMSSD), SCL, SCRrate, PERCLOS, IrisVar] :contentReference[oaicite:3]{index=3}

---

###  Modelo de inferencia

Se utiliza un **observador de Kalman con estado aumentado** para compensar el retraso de las mediciones.

Estado:

x_k = [a_k, o_k, d_k]

- a_k → activación autonómica  
- o_k → inestabilidad ocular  
- d_k → deriva global  

El sistema estima:

- Proyección Kalman: J_k
- Distancia Mahalanobis: D_k

Índice final:

I_k = α·J_k + β·(D_k / (D_k + 3)) :contentReference[oaicite:4]{index=4}

---

## 3. Protocolo experimental

El sistema incluye bloques cognitivos diseñados para inducir carga atencional:

### Bloque 3 – Memoria operativa
- Presentación de números aleatorios
- Identificación de paridad del estímulo dominante
- Registro de:
  - RT (tiempo de reacción)
  - exactitud
  - omisiones :contentReference[oaicite:5]{index=5}

---

## 4. Flujo completo

1. Calibración basal (fisiológica y ocular)
2. Registro multimodal en tiempo real
3. Extracción de características
4. Estimación de estado latente (Kalman)
5. Generación de índice de deriva
6. Almacenamiento estructurado por sesión

---

## 5. Estructura de datos

Cada sesión genera:
/outputs/
/subject_id/
/session_id/
session_meta.json
events.csv
ocular_all_samples.csv
physio_features_all.csv
trials_long.csv
blocks_summary.csv
/bloques/


El sistema mantiene trazabilidad completa:

- Tiempo absoluto (pc_timestamp)
- Tiempo relativo (master_rel_s)
- Identificador de sesión y sujeto :contentReference[oaicite:6]{index=6}

---

## 6. Tecnologías utilizadas

- Python 3.11
- OpenCV
- MediaPipe
- NumPy / SciPy
- PySide6 (interfaz)
- NI-DAQmx (adquisición)
- Pandas

---

## 7. Consideraciones técnicas

- No se asume alineación directa entre señales → sincronización por timestamp
- Manejo explícito de datos faltantes
- Validación de calidad por frame (ocular)
- Modelado robusto frente a ruido fisiológico

---

## 8. Limitaciones

- Dependencia de iluminación para tracking ocular
- Sensibilidad a movimiento de cabeza (>15°)
- ECG y EDA requieren correcta instrumentación
- No es un sistema clínico diagnóstico

---

## 9. Aplicaciones

- Evaluación de atención sostenida
- Estudios cognitivos experimentales
- Interfaces humano-máquina adaptativas
- Investigación en neuroingeniería

---

## 10. Futuro trabajo

- Integración de EEG en tiempo real
- Modelos adaptativos personalizados
- Reducción de latencia (<30 s)
- Clasificación automática de estados (ML)

---

## 11. Autoría

Proyecto desarrollado como parte de Ingeniería Biomédica  
CUCEI – Universidad de Guadalajara

---

