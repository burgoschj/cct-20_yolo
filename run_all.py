#!/usr/bin/env python
"""Punto de entrada único.

    python run_all.py            # pipeline completo (descarga CCT-20, entrena, evalúa, monitorea)
    python run_all.py --smoke    # verificación rápida con datos sintéticos (CPU, minutos)
"""
import argparse
from pathlib import Path

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser(description='Pipeline CCT-20 + YOLOv8 de extremo a extremo.')
    p.add_argument('--smoke', action='store_true', help='prueba rápida con datos sintéticos y un modelo diminuto')
    p.add_argument('--config', default=None, help='YAML de configuración (por defecto configs/default.yaml o configs/smoke.yaml)')
    p.add_argument('--data-dir', default=None, help='dónde descargar/guardar los datos (por defecto ./data)')
    p.add_argument('--out-dir', default=None, help='dónde escribir resultados (por defecto ./outputs). En Colab: una carpeta de Drive')
    p.add_argument('--until', default=None, help='detenerse tras una etapa (p. ej. s02 = solo datos + entrenamiento)')
    p.add_argument('--set', dest='overrides', action='append', metavar='CLAVE=VALOR', help='sobrescribe un parámetro, p. ej. --set epochs=30')
    a = p.parse_args()
    from cct20.runner import run
    cfg = a.config or str(HERE / 'configs' / ('smoke.yaml' if a.smoke else 'default.yaml'))
    data = a.data_dir or str(HERE / ('data_smoke' if a.smoke else 'data'))
    out = a.out_dir or str(HERE / ('outputs_smoke' if a.smoke else 'outputs'))
    run(cfg, data, out, until=a.until, overrides=a.overrides, smoke=a.smoke)


if __name__ == '__main__':
    main()
