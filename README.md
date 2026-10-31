# Monitoreo de fauna con YOLOv8 en Caltech Camera Traps (CCT-20)

Detección de fauna en fotos de cámaras trampa con **YOLOv8s**, evaluada con rigor: mAP por especie y efecto del
desbalance, generalización a cámaras nunca vistas (cis vs trans) con Grad-CAM, robustez a oclusión y tamaño,
un módulo de monitoreo basado en reglas con demo en una cámara no vista.

## Ejecución de extremo a extremo (un único comando)

```bash
pip install -r requirements.txt     # 1) entorno
python run_all.py                   # 2) TODO el proyecto: descarga → entrena → evalúa → monitorea → empaqueta
```

`python run_all.py` descarga CCT-20 (~6 GB) desde LILA, construye el dataset YOLO, entrena, evalúa y
deja en `outputs/` las tablas, figuras, el reporte de monitoreo y un `.zip` con los entregables.

**¿Quieres comprobar que el repositorio funciona antes de gastar horas de GPU?**

```bash
python run_all.py --smoke           # ~1-2 min en CPU, datos sintéticos, sin descargas
```

`--smoke` genera un mini-dataset sintético con el mismo formato que CCT-20 y corre **las 9 etapas completas** con un
modelo diminuto. Verifica que el código corre de punta a punta; sus métricas (≈0) no tienen significado científico.
También: `python tests/test_smoke.py`.

## Entorno

| Archivo | Para qué |
|---|---|
| `requirements.txt` | Especificación principal (rangos compatibles; `ultralytics==8.4.190` fijo) |
| `requirements-lock.txt` | Versiones exactas con las que se probó el pipeline (Python 3.13) |
| `environment.yml` | Alternativa con conda: `conda env create -f environment.yml` |

Python 3.10–3.13. Se necesita **GPU** para el entrenamiento completo (se usó una T4 de Colab: ≈2.4 h para 39 épocas
con parada temprana). Sin GPU el pipeline corre pero el entrenamiento completo sería impracticable.
PyTorch no está fijado: en Colab ya viene instalado con CUDA (se usó 2.11.0+cu130); en local instala la
build adecuada desde <https://pytorch.org> antes de `pip install -r requirements.txt`.

## Ejecutar en Google Colab (recomendado para el entrenamiento completo)

1. *Entorno de ejecución → Cambiar tipo → GPU T4*.
2. En una celda:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/burgoschj/cct-20_yolo.git
%cd <TU_REPO>
!pip -q install -r requirements.txt
!python run_all.py --data-dir /content/cct20_data --out-dir /content/drive/MyDrive/cct20_yolo
```

- `--data-dir` va en el disco local rápido de Colab (los datos se vuelven a descargar si la sesión se pierde).
- `--out-dir` va en **Drive** para que entrenamiento, predicciones y resultados sobrevivan a una caída de sesión.
- **Si la sesión se cae, vuelve a ejecutar el mismo comando**: reanuda el entrenamiento desde `last.pt` y reutiliza
  dataset, pesos y predicciones ya guardados.
- Si ya se tiene la carpeta `cct20_yolo` en Drive de la corrida previa en el cuaderno, apuntar `--out-dir` ahí
  reutiliza el modelo entrenado (E0) y las predicciones: solo se recalcula la evaluación.

## Opciones

```bash
python run_all.py --until s02                      # solo datos + entrenamiento
python run_all.py --set epochs=30 --set batch=16   # sobrescribir hiperparámetros sin editar archivos
python run_all.py --config configs/mi.yaml         # otra configuración
python run_all.py --data-dir D --out-dir O         # rutas de datos y resultados
```

Toda la configuración está en `configs/default.yaml`.

## Qué hace cada etapa y dónde está cada entregable

| Etapa | Archivo | Entregable | Salidas principales (`outputs/`) |
|---|---|---|---|
| s00 | `cct20/stages/s00_setup.py` | Configuración, semillas, rutas | — |
| s01 | `s01_data.py` | Datos: descarga, EDA, dataset YOLO, verificación de etiquetas | `figuras/01_eda.png`, `02_verificacion_etiquetas.png` |
| s02 | `s02_train.py` | **1. Modelo YOLO** y decisiones de entrenamiento | `runs/E0_base/` (`decisiones_entrenamiento.json`, `weights/best.pt`), `figuras/03_curvas_entrenamiento.png` |
| s03 | `s03_eval.py` | **2. mAP@0.5 y @0.5:0.95 por especie** + efecto del desbalance | `tablas/03_*`, `04_*`, `figuras/04_desbalance_vs_ap.png` |
| s04 | `s04_generalization.py` | **3.** Brecha cis/trans, AP por ubicación | `tablas/07_*`, `09_*`, `10_*`, `figuras/05_*` |
| s05 | `s05_gradcam.py` | **3.** Grad-CAM + prueba del animal enmascarado | `tablas/11_*`, `12_*`, `13_*`, `figuras/06_*`, `07_*` |
| s06 | `s06_robustness.py` | **4. Robustez** a oclusión y tamaño | `tablas/14_*`–`16_*`, `figuras/08_*`, `09_*` |
| s07 | `s07_monitoring.py` | **5. Módulo de monitoreo** (reglas R1–R5) y demo en cámara no vista | `reporte_monitoreo.xlsx`, `tablas/18_*`–`22_*`, `figuras/10_*` |
| s08 | `s08_package.py` | Paquete final | `entregables_resultados.zip`, `RESUMEN.md` |


## Decisiones metodológicas clave

- **Validación = `trans_val`**. `cis_val` comparte 224 secuencias con `train`, así que no sirve para elegir checkpoint sin filtrarse información.
- **Fondos**: el `train` oficial no tiene imágenes vacías; se añaden 995 vacías de `cis_val` como fondos. Las 7 824 imágenes sin etiqueta se excluyen.
- **mAP por subgrupo** con `pycocotools` (cajas fuera del grupo se marcan `iscrowd=1`); sus valores son 0.01–0.03 menores que los de Ultralytics por usar `conf=0.01` y `max_det=50` (en vez de 0.001 y 300). Los números del informe indican cuál se usa en cada caso.
- **Reglas del monitoreo calibradas en `trans_val`**, nunca en test; la demo usa una cámara de `trans_test` ausente del entrenamiento.
- Semilla fija (42). `cudnn.benchmark=False` evita un pico de ~14 GB de GPU en la primera inferencia (importante en T4).

## Estructura

```
run_all.py            # único punto de entrada
configs/              # default.yaml (completo) · smoke.yaml (prueba rápida)
cct20/runner.py       # ejecuta las etapas en orden con espacio de nombres compartido
cct20/stages/         # s00 … s08: el código del proyecto, por etapa
cct20/synthetic.py    # dataset sintético para --smoke
tests/test_smoke.py   # prueba de extremo a extremo
notebooks/            # versión original en cuaderno de Colab
```


## Referencias

Beery et al. (2018), *Recognition in Terra Incognita*, ECCV · Jocher et al., Ultralytics YOLO · Selvaraju et al. (2017), Grad-CAM · LILA BC: <https://lila.science/datasets/caltech-camera-traps>
