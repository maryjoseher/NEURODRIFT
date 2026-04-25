Paquete corregido para adquisición con 2 NI independientes.

Conexión por defecto:
- ECG: Dev1/ai0
- EDA: Dev2/ai1

Cambios clave:
- physio_ni.py usa 2 tareas NI independientes, una por dispositivo.
- physio_runtime.py es RAW real: sin HRV, sin SCR, sin picos.
- app.py conserva consola, tracking ocular y bloques, y guarda el CSV crudo dentro de raw_physio/ al usar Guardar Sesion.

Si tus nombres en NI MAX no coinciden, edita en app.py la inicialización de self.physio_cfg.
