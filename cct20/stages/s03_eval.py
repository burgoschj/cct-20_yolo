# Etapa s03_eval: mAP por especie y efecto del desbalance
# Se ejecuta con exec() en un espacio de nombres compartido (ver cct20/runner.py).

# ----------------------------------------------------------------------
# [eval_fn]
# ----------------------------------------------------------------------
def ultra_val(weights, yaml_path, tag, imgsz=None, plots=False):
    """Evalúa con Ultralytics. Devuelve métricas globales y DataFrame por especie."""
    model = YOLO(str(weights))
    m = model.val(data=str(yaml_path), imgsz=imgsz or CFG.imgsz, batch=CFG.batch, conf=0.001, iou=CFG.iou_nms,
                  device=DEVICE, **HALF_KW, plots=plots, project=str(RUNS / 'eval'), name=tag, exist_ok=True,
                  verbose=False, workers=CFG.workers)
    b = m.box
    glob = dict(P=float(b.mp), R=float(b.mr), mAP50=float(b.map50), mAP50_95=float(b.map))
    per = pd.DataFrame({'species': [class_names[int(i)] for i in b.ap_class_index],
                        'AP50': b.all_ap[:, 0], 'AP50_95': b.all_ap.mean(1), 'P': b.p, 'R': b.r})
    return glob, per, Path(m.save_dir)

def evaluate_all(weights, ds_, exp, imgsz=None):
    G, PER = {}, {}
    for s in ['cis_val', 'trans_val', 'cis_test', 'trans_test', 'test_all']:
        G[s], PER[s], sd = ultra_val(weights, ds_.yaml[s], f'{exp}_{s}', imgsz, plots=(s == 'test_all' and CFG.train_plots))
        if s == 'test_all': SAVE_DIR = sd
    return G, PER, SAVE_DIR


# ----------------------------------------------------------------------
# [eval_e0]
# ----------------------------------------------------------------------
G, PER, SAVE_DIR = evaluate_all(BEST, ds, 'E0')
glob_tab = pd.DataFrame(G).T[['mAP50', 'mAP50_95', 'P', 'R']].round(4)
glob_tab.to_csv(TABLES / '03_metricas_globales_E0.csv'); glob_tab


# ----------------------------------------------------------------------
# [per_species]
# ----------------------------------------------------------------------
# Conteos de ejemplos por especie
def count_boxes(split_list):
    d = ds.df[ds.df.split.isin(split_list)]
    c = Counter(b[0] for bl in d.boxes for b in bl)
    return pd.Series({n: c.get(i, 0) for i, n in enumerate(class_names)})
def count_imgs(split_list):
    d = ds.df[ds.df.split.isin(split_list)]
    c = Counter(i for bl in d.boxes for i in {b[0] for b in bl})
    return pd.Series({n: c.get(i, 0) for i, n in enumerate(class_names)})

sp = pd.DataFrame({'n_train_cajas': count_boxes(['train']), 'n_train_imgs': count_imgs(['train']),
                   'n_test_cis': count_boxes(['cis_test']), 'n_test_trans': count_boxes(['trans_test'])})
for s in ['cis_test', 'trans_test', 'test_all']:
    t = PER[s].set_index('species')[['AP50', 'AP50_95']].add_suffix('_' + s.replace('_test', '').replace('test_all', 'todo'))
    sp = sp.join(t)
sp['n_test_todo'] = sp.n_test_cis + sp.n_test_trans
sp['aviso'] = np.where(sp.n_test_todo < 30, '⚠️ n_test<30 (ruidoso)', '')
sp = sp.sort_values('n_train_cajas', ascending=False)
sp.round(3).to_csv(TABLES / '04_ap_por_especie_E0.csv')
display(sp.round(3))

# ---- Efecto del desbalance ----
from scipy.stats import spearmanr
v = sp.dropna(subset=['AP50_95_todo']).copy()
rho, pval = spearmanr(np.log10(v.n_train_cajas.clip(lower=1)), v.AP50_95_todo)
v['grupo'] = pd.qcut(v.n_train_cajas.rank(method='first'), 3, labels=['rara', 'intermedia', 'común'])
grp = v.groupby('grupo').agg(especies=('AP50_95_todo', 'size'), n_train_med=('n_train_cajas', 'median'),
                             AP50=('AP50_todo', 'mean'), AP50_95=('AP50_95_todo', 'mean')).round(3)
grp.to_csv(TABLES / '05_ap_por_grupo_de_frecuencia.csv')
print(f'Spearman(log n_train, AP50-95) = {rho:.2f}  (p = {pval:.3f}, n = {len(v)} especies)')
display(grp)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].scatter(v.n_train_cajas, v.AP50_95_todo, s=60, c='#3b7dd8')
for n, r in v.iterrows(): ax[0].annotate(n, (r.n_train_cajas, r.AP50_95_todo), fontsize=8, xytext=(4, 3), textcoords='offset points')
ax[0].set_xscale('log'); ax[0].set_xlabel('n.º de cajas de entrenamiento (log)'); ax[0].set_ylabel('AP@0.5:0.95 (test)')
ax[0].set_title(f'Desbalance vs desempeño (Spearman ρ={rho:.2f})')
o = v.sort_values('n_train_cajas'); ax[1].barh(o.index, o.AP50_95_todo, color='#3b7dd8')
ax[1].set_xlabel('AP@0.5:0.95 (test) — ordenado de rara (arriba) a común (abajo)'); ax[1].set_title('AP por especie')
plt.tight_layout(); plt.savefig(FIGS / '04_desbalance_vs_ap.png', dpi=130); plt.close('all')


# ----------------------------------------------------------------------
# [confmat]
# ----------------------------------------------------------------------
for f in ['confusion_matrix_normalized.png', 'confusion_matrix.png']:
    p = SAVE_DIR / f
    if p.exists():
        shutil.copy2(p, FIGS / ('05b_' + f)); print('Matriz de confusión →', FIGS / ('05b_' + f)); break


# ----------------------------------------------------------------------
# [optional_exps]
# ----------------------------------------------------------------------
RUN_E1, RUN_E2, RUN_E3 = CFG.run_e1, CFG.run_e2, CFG.run_e3

def make_variant(ds_, kind='rfs', t=0.05, cap=4):
    """Crea train_rfs/ con hardlinks (no duplica disco): repite más las imágenes con especies raras."""
    src = ds_.df[ds_.df.split == 'train']
    vname = f'train_{kind}'
    for sub_ in ('images', 'labels'): (ds_.root / sub_ / vname).mkdir(parents=True, exist_ok=True)
    f = Counter(c for bl in src.boxes for c in {b[0] for b in bl})
    f = {c: n / len(src) for c, n in f.items()}
    rc = {c: max(1.0, math.sqrt(t / fc)) for c, fc in f.items()}
    print('repeat factor por especie:', {class_names[c]: round(v, 2) for c, v in sorted(rc.items())})
    n = 0
    for r in src.itertuples():
        cs = {b[0] for b in r.boxes}
        reps = 1 if not cs else min(cap, math.ceil(max(rc[c] for c in cs)))
        for k in range(reps):
            stem = Path(r.fname).stem + (f'__r{k}' if k else '')
            di, dl = ds_.root / 'images' / vname / (stem + '.jpg'), ds_.root / 'labels' / vname / (stem + '.txt')
            if not di.exists():
                try: os.link(r.ypath, di)
                except OSError: shutil.copy2(r.ypath, di)
            shutil.copy2(ds_.root / 'labels' / 'train' / (Path(r.fname).stem + '.txt'), dl); n += 1
    y = ds_.root / f'data_{vname}.yaml'
    tr = [f'images/{vname}'] + (['images/train_bg'] if ds_.n_bg else [])          # los fondos se mantienen
    yaml.safe_dump(dict(path=str(ds_.root), train=tr, val=[f'images/{s}' for s in CFG.val_splits],
                        names={i: nm for i, nm in enumerate(class_names)}), open(y, 'w'), sort_keys=False)
    print(f'{vname}: {n} imágenes de entrenamiento con animal (original: {len(src)}) + {ds_.n_bg} fondos'); return y

EXPS = {'E0_base': (BEST, ds, CFG.imgsz)}
if RUN_E1:
    EXPS['E1_sin_fondos'] = (train_experiment('E1_sin_fondos', ds.yaml['train_only']), ds, CFG.imgsz)
if RUN_E2:
    EXPS['E2_rfs'] = (train_experiment('E2_rfs', make_variant(ds)), ds, CFG.imgsz)
if RUN_E3:
    ds960 = build_dataset('s960', 960)
    EXPS['E3_960'] = (train_experiment('E3_960', ds960.yaml['train'], imgsz=960, batch=16), ds960, 960)

if len(EXPS) > 1:
    rows = []
    for k, (w, d_, sz) in EXPS.items():
        for s in ['cis_test', 'trans_test', 'test_all']:
            g, _, _ = ultra_val(w, d_.yaml[s], f'{k}_{s}', sz)
            rows.append(dict(exp=k, split=s, **{m: round(v, 4) for m, v in g.items()}))
    cmp_tab = pd.DataFrame(rows).pivot(index='exp', columns='split', values=['mAP50', 'mAP50_95'])
    cmp_tab.to_csv(TABLES / '06_comparacion_experimentos.csv'); display(cmp_tab)
else:
    print('Solo E0 activo. Activa RUN_E1/E2/E3 para comparar.')

