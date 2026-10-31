# Etapa s00_setup: Configuración, semillas y rutas
# Se ejecuta con exec() en un espacio de nombres compartido (ver cct20/runner.py).

# ----------------------------------------------------------------------
# [config]
# ----------------------------------------------------------------------
import os, sys, json, math, random, shutil, time, zipfile, tarfile, subprocess, warnings, gc
from pathlib import Path
from collections import Counter, defaultdict
from types import SimpleNamespace as NS
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')                       # sin pantalla: las figuras solo se guardan a PNG
import matplotlib.pyplot as plt
import cv2, torch, yaml
from PIL import Image
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50); pd.set_option('display.width', 200)

# `RUN` lo inyecta cct20/runner.py: rutas (data_dir, out_dir) y el bloque `cfg` del YAML.
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False      # evita un pico de ~14 GB de memoria GPU en la primera inferencia

ROOT = Path(RUN['data_dir']).resolve()      # datos originales + dataset en formato YOLO
OUT = Path(RUN['out_dir']).resolve()        # resultados: corridas, tablas, figuras, predicciones
RAW    = ROOT / 'cct_raw'
RUNS   = OUT / 'runs'
TABLES = OUT / 'tablas'
FIGS   = OUT / 'figuras'
PREDS  = OUT / 'preds'
for d in (RAW, RUNS, TABLES, FIGS, PREDS): d.mkdir(parents=True, exist_ok=True)

_defaults = dict(model='yolov8s.pt', imgsz=640, max_side=640, epochs=50, batch=32, workers=2, patience=15,
                 conf_pred=0.01, iou_nms=0.7, max_det=50, conf_op=0.25, gradcam_layers=(4, 6, 9),
                 bg_from_cis_val=True, val_splits=('trans_val',), non_animal=('car',),
                 min_boxes=30, n_cam=150, n_masked=200, train_plots=True, run_e1=False, run_e2=False, run_e3=False)
_user = {k: (tuple(v) if isinstance(v, list) else v) for k, v in RUN['cfg'].items()}
_unknown = set(_user) - set(_defaults)
assert not _unknown, f'Claves desconocidas en el YAML: {_unknown}'
CFG = NS(**{**_defaults, **_user})
DEVICE   = 0 if torch.cuda.is_available() else 'cpu'
USE_HALF = bool(torch.cuda.is_available())
HALF_KW  = dict(quantize=16) if USE_HALF else {}      # FP16 solo si hay GPU
print('Dispositivo:', DEVICE, '| Torch', torch.__version__, '| datos:', ROOT, '| salidas:', OUT)
print('CFG =', vars(CFG))

