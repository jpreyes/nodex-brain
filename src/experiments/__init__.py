"""Experimentos de investigación. NADA de aquí entra en la ruta de producción.

Producción vive en `src/train_sft.py` y no importa nada de este paquete.
Los experimentos sí importan de producción — a propósito: comparten modelo, datos y
preprocesado, y solo cambian la variable bajo estudio. Duplicar el preprocesado haría que
el Δ dejara de ser atribuible a la variable.
"""
