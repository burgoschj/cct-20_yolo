# Etapa s08_package: Paquete final y resumen
# Se ejecuta con exec() en un espacio de nombres compartido (ver cct20/runner.py).

# ----------------------------------------------------------------------
# [package]
# ----------------------------------------------------------------------
files = sorted([p for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.csv', '.png', '.xlsx', '.json', '.yaml'} and 'preds' not in p.parts])
print(f'{len(files)} archivos de resultados:'); [print(' ·', p.relative_to(OUT)) for p in files[:80]]
zpath = OUT / 'entregables_resultados.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p, p.relative_to(OUT))
    if (RUNS / 'E0_base' / 'weights' / 'best.pt').exists(): z.write(RUNS / 'E0_base' / 'weights' / 'best.pt', 'E0_best.pt')
print('\nPaquete:', zpath, f'({zpath.stat().st_size/1e6:.1f} MB)')


# ----------------------------------------------------------------------
# [resumen]
# ----------------------------------------------------------------------
def _tab(p):
    t = pd.read_csv(p)
    if t.columns[0].startswith('Unnamed'): t = t.drop(columns=t.columns[0]) if pd.api.types.is_numeric_dtype(t.iloc[:, 0]) else t.rename(columns={t.columns[0]: 'fila'})
    t = t.round(3)
    return t.to_markdown(index=False) if _has_tabulate else t.to_string(index=False)
try:
    import tabulate; _has_tabulate = True
except ImportError:
    _has_tabulate = False

# Resumen legible de lo que produjo el pipeline
lines = ['# Resumen de la ejecución', '']
for fname, title in [('03_metricas_globales_E0.csv', 'Métricas globales (E0)'), ('07_brecha_cis_trans.csv', 'Brecha cis → trans'),
                     ('13_prueba_animal_enmascarado.csv', 'Prueba del animal enmascarado'), ('22_monitoreo_modelo_vs_real_por_especie.csv', 'Monitoreo: modelo vs real')]:
    p = TABLES / fname
    if p.exists():
        lines += [f'## {title}', '', _tab(p), '']
(OUT / 'RESUMEN.md').write_text('\n'.join(lines), encoding='utf-8')
print('Resumen →', OUT / 'RESUMEN.md')

