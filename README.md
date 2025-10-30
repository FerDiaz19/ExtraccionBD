# LogisticRegression - Instrucciones de uso

Este repositorio contiene una API Flask y una página HTML exportada desde un cuaderno (Notebook) que muestran un modelo de Regresión Logística entrenado (pickle), un scaler y una interfaz para hacer predicciones y recomendaciones en tiempo real.

Este README explica cómo descargar/instalar dependencias, preparar el entorno en Windows (PowerShell), ejecutar el servidor y las pruebas, y cómo usar los endpoints principales.

## Requisitos

- Python 3.8+ (preferible 3.9/3.10). Asegúrate de tener instalado Python y `pip`.
- Windows PowerShell (las instrucciones usan PowerShell). En otros sistemas las órdenes son análogas.
- Acceso a Internet para instalar dependencias (solo la primera vez).

Dependencias listadas en `requirements.txt` (ya incluidas en el repo):

- flask
- flask-cors
- joblib
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

## Descargar o clonar el repositorio

Si tienes el repositorio remoto (GitHub), clónalo; si no, descarga el ZIP y extrae la carpeta:

```powershell
git clone <URL-de-tu-repo>
cd "c:\Users\fdiaz\OneDrive - Universidad Tecnológica de Tijuana\9no\BD\LogisticRegression"
```

Si ya trabajas localmente en la carpeta, sitúate en ella:

```powershell
cd "C:\Users\fdiaz\OneDrive - Universidad Tecnológica de Tijuana\9no\BD\LogisticRegression"
```

## Crear y activar un entorno virtual (recomendado)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Ahora el prompt mostrará (.venv) si todo OK
```

Si PowerShell bloquea la ejecución de scripts, habilita temporalmente (ejecutar como Administrador si es necesario):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Instalar dependencias

Con el entorno virtual activado, instala los paquetes:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Archivos importantes

- `app.py` — servidor Flask que expone endpoints para estado, muestras y recomendaciones.
- `Documentacion_Modelo_Regresion_Logistica_dark.html` — página HTML exportada del notebook (UI). Incluye un script `js/predict.js` que consume la API.
- `js/predict.js` — lógica cliente para poblar la página, hacer predicciones y manejar uploads.
- `modelo_regresion_logistica.pkl`, `scaler.pkl` — artefactos del modelo. Si no están presentes, el servidor seguirá arrancando pero muchas funcionalidades estarán limitadas.
- `sample_rows.json` — filas de ejemplo usadas por la UI y por el endpoint `/recommend_batch`.
- `plots/` (o archivos `confusion.png`, `roc.png`, `decision.png`, `sample_plot.png`) — imágenes generadas por el servidor cuando se solicita.
- `model_metrics.json` — métricas del modelo generadas por la función que crea plots.

## Generar/obtener los pickles del modelo

El repo asume que `modelo_regresion_logistica.pkl` y `scaler.pkl` existen. Si no los tienes, puedes:

1. Reproducir el notebook Lab04 (si lo tienes) y ejecutar las celdas que guardan con `joblib.dump(...)`.
2. Pedir al autor que comparta los `.pkl` y colocarlos en la carpeta del proyecto.

Si no hay modelo, el servidor seguirá proporcionando endpoints que devuelven mensajes explicativos, y la UI mostrará que no hay modelo cargado.

## Ejecutar el servidor

Con el entorno activado y desde la carpeta del proyecto:

```powershell
py -3 app.py
# o: python app.py
```

El servidor corre por defecto en `http://127.0.0.1:5000/`. Abre esa URL en tu navegador para ver la página `Documentacion_Modelo_Regresion_Logistica_dark.html` servida por Flask.

> Nota: `app.py` desactiva el autoreloader (use_reloader=False) para evitar comportamientos en multiproceso en algunas pruebas.

## Endpoints útiles

- GET `/` — sirve `Documentacion_Modelo_Regresion_Logistica_dark.html`.
- GET `/status` — información sobre si el modelo y scaler están cargados y metadatos (coeficientes, media del scaler).
- GET `/sample_rows?n=5` — devuelve hasta `n` filas de ejemplo (JSON list of dicts).
- GET `/pipeline` — información del pipeline (columnas y pasos).
- GET `/metrics` — lee `model_metrics.json` si está disponible.
- POST `/upload` — subir nuevos archivos `model` y/o `scaler` como multipart/form-data. Ejemplo con PowerShell:

```powershell
$form = @{ model = Get-Item .\modelo_regresion_logistica.pkl; scaler = Get-Item .\scaler.pkl }
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/upload' -Method Post -Form $form
```

- POST `/reload` — fuerza recarga de pickles si han cambiado en disco.
- POST or GET `/generate_plots` — genera PNGs (`confusion.png`,`roc.png`,`decision.png`,`sample_plot.png`) y escribe `model_metrics.json`.
- GET `/plots/<filename>` — sirve las imágenes generadas (allowed: confusion.png, roc.png, decision.png, sample_plot.png).
- POST `/recommend` — uso en tiempo real: JSON `{"age": 30, "salary": 50000}` → devuelve `prediction`, `probability`, `suggestion`.
- GET `/recommend_batch` — usa `sample_rows.json` para devolver recomendaciones en lote (JSON con array `results`).

Ejemplo de prueba rápida (PowerShell):

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/status' -Method Get | ConvertTo-Json -Depth 5
```

Ejemplo de `recommend`:

```powershell
$body = @{ age = 30; salary = 50000 } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/recommend' -Method Post -Body $body -ContentType 'application/json'
```

## Ver la UI exportada (notebook HTML)

La página `Documentacion_Modelo_Regresion_Logistica_dark.html` es servida por Flask en `/`. Al abrirla en el navegador, el JS embebido consultará los endpoints y poblará las secciones (muestras, métricas, recomendaciones, gráficos).

Si la UI no muestra resultados de inmediato:

- Asegúrate que el servidor esté corriendo y que `/recommend_batch` y `/sample_rows` devuelvan 200.
- Revisa la consola del navegador (F12) por errores JS o CORS.

## Pruebas unitarias

Hay un archivo `test_predict.py` en el repositorio. Puedes ejecutar con `pytest` (instálalo si no está):

```powershell
pip install pytest
pytest -q test_predict.py
```

## Troubleshooting / Preguntas frecuentes

- Si recibes 500 en `/recommend` -> comprueba que `modelo_regresion_logistica.pkl` y `scaler.pkl` estén en la carpeta del proyecto y sean válidos.
- Si `/recommend_batch` devuelve 404 -> revisa que `sample_rows.json` exista y sea un JSON válido (lista de objetos). Hay un ejemplo `sample_rows.json` incluido.
- Las imágenes pueden mostrarse como 304 (Not Modified) por el navegador; para forzar recarga puedes pedir `GET /generate_plots` y luego recargar la página.

## Seguridad y notas finales

- Este servidor está pensado para desarrollo/demo local. No debe exponerse a internet sin agregar autenticación y medidas de seguridad.
- Si compartes los pickles, evita exponer datos sensibles. Los archivos `.pkl` pueden ejecutar código si se cargan de fuentes no confiables.

Si quieres, puedo añadir instrucciones para generar los `.pkl` desde el notebook Lab04 (pasos concretos), o crear un pequeño script `make_model.py` que genere un modelo de ejemplo si te interesa.

---

Fecha: 2025-10-30
