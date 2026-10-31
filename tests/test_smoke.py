"""Prueba de extremo a extremo con datos sintéticos:  python -m pytest -q   (o:  python tests/test_smoke.py)"""
import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ['tablas/03_metricas_globales_E0.csv', 'tablas/07_brecha_cis_trans.csv', 'tablas/09_ap_por_ubicacion.csv',
            'tablas/12_gradcam_resumen.csv', 'tablas/13_prueba_animal_enmascarado.csv', 'tablas/15_robustez_por_grupo.csv',
            'tablas/18_calibracion_umbrales_val.csv', 'tablas/22_monitoreo_modelo_vs_real_por_especie.csv',
            'figuras/01_eda.png', 'figuras/06_gradcam_galeria.png', 'figuras/09_robustez.png', 'figuras/10_monitoreo_demo.png',
            'reporte_monitoreo.xlsx', 'entregables_resultados.zip', 'RESUMEN.md', 'run_log.json']


def test_pipeline_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, str(ROOT / 'run_all.py'), '--smoke', '--data-dir', f'{tmp}/data', '--out-dir', f'{tmp}/out'],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        missing = [f for f in EXPECTED if not (Path(tmp) / 'out' / f).exists()]
        assert not missing, f'Faltan salidas: {missing}'


if __name__ == '__main__':
    test_pipeline_smoke(); print('OK')
