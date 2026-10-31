"""Ejecuta las etapas del pipeline en orden, compartiendo un único espacio de nombres (igual que las celdas del cuaderno)."""
import json, sys, time, traceback
from pathlib import Path
import yaml

STAGES_DIR = Path(__file__).parent / 'stages'
STAGES = ['s00_setup', 's01_data', 's02_train', 's03_eval', 's04_generalization',
          's05_gradcam', 's06_robustness', 's07_monitoring', 's08_package']


def _display(x, *a, **k):
    print(x.to_string(max_rows=40) if hasattr(x, 'to_string') else x)


def _parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def run(config, data_dir, out_dir, until=None, overrides=None, smoke=False):
    cfg_file = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    cfg = {**cfg_file.get('cfg', {}), **_parse_overrides(overrides)}
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    data_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        from .synthetic import make_synthetic
        raw = data_dir / 'cct_raw'
        if not (raw / 'annotations').exists():
            print('[smoke] generando dataset sintético en', raw)
            make_synthetic(raw)

    names = STAGES
    if until:
        hit = [i for i, s in enumerate(STAGES) if s.startswith(until)]
        if not hit: sys.exit(f'Etapa desconocida: {until}. Opciones: {STAGES}')
        names = STAGES[:hit[0] + 1]

    ns = {'__name__': '__main__', 'RUN': dict(cfg=cfg, data_dir=str(data_dir), out_dir=str(out_dir), smoke=smoke), 'display': _display}
    timings = {}
    t_all = time.time()
    for s in names:
        print(f'\n{"=" * 78}\n>>> {s}\n{"=" * 78}', flush=True)
        t = time.time()
        code = (STAGES_DIR / f'{s}.py').read_text(encoding='utf-8')
        try:
            exec(compile(code, str(STAGES_DIR / f'{s}.py'), 'exec'), ns)
        except Exception:
            traceback.print_exc()
            print(f'\n✖ FALLÓ la etapa {s}. Vuelve a ejecutar el MISMO comando: reanuda sin repetir lo ya hecho '
                  f'(dataset, entrenamiento y predicciones quedan en caché).', file=sys.stderr)
            sys.exit(1)
        timings[s] = round(time.time() - t, 1)
        print(f'-- {s} OK en {timings[s]} s', flush=True)
    (out_dir / 'run_log.json').write_text(json.dumps(dict(config=str(config), cfg=cfg, smoke=smoke, segundos=timings,
                                                         total_s=round(time.time() - t_all, 1)), indent=2), encoding='utf-8')
    print(f'\n✔ PIPELINE COMPLETO en {time.time() - t_all:.0f} s. Resultados en: {out_dir}')
