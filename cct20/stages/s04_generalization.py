# Etapa s04_generalization: Predicciones, brecha cis/trans, AP por ubicación
# Se ejecuta con exec() en un espacio de nombres compartido (ver cct20/runner.py).

# ----------------------------------------------------------------------
# [pred_utils]
# ----------------------------------------------------------------------
import contextlib, io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
TDEV = torch.device('cuda:0' if DEVICE != 'cpu' else 'cpu')

def predict_split(weights, d, tag, imgsz=None, chunk=1000, bs=16):
    """Predice en todas las imágenes de d por trozos (guarda parciales; no acumula memoria GPU).
    Devuelve array Nx7: [uid, x, y, w, h, score, clase+1] (px)."""
    f = PREDS / f'{tag}.npy'
    if f.exists(): return np.load(f)
    model = YOLO(str(weights)); uid_by = dict(zip(d.fname, d.uid))
    paths = list(d.ypath); parts = []
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    nb = 0
    for i in tqdm(range(0, len(paths), chunk), desc=f'predict {tag}'):
        part = PREDS / f'{tag}_part{i // chunk:03d}.npy'
        if part.exists(): parts.append(np.load(part)); continue
        rows = []
        for j in range(i, min(i + chunk, len(paths)), bs):
            with torch.inference_mode():
                rs = model.predict(source=paths[j:j + bs], conf=CFG.conf_pred, iou=CFG.iou_nms, imgsz=imgsz or CFG.imgsz,
                                   batch=bs, max_det=CFG.max_det, device=DEVICE, verbose=False, **HALF_KW)
            for r in rs:
                b = r.boxes
                if b is None or len(b) == 0: continue
                xyxy = b.xyxy.cpu().numpy(); n = len(xyxy)
                a = np.empty((n, 7), dtype=np.float32)
                a[:, 0] = uid_by[Path(r.path).name]
                a[:, 1:3] = xyxy[:, :2]; a[:, 3:5] = xyxy[:, 2:] - xyxy[:, :2]
                a[:, 5] = b.conf.cpu().numpy(); a[:, 6] = b.cls.cpu().numpy() + 1
                rows.append(a)
            del rs; nb += 1
            if torch.cuda.is_available() and nb % 20 == 0 and torch.cuda.memory_allocated() / 1e9 > 6:
                raise RuntimeError('La memoria GPU crece durante la predicción (>6 GB). Reinicia y vuelve a ejecutar: continúa desde los parciales.')
        arr = np.concatenate(rows) if rows else np.zeros((0, 7), np.float32)
        np.save(part, arr); parts.append(arr)
        del rows; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    arr = np.concatenate(parts).astype(np.float64); np.save(f, arr)
    for p in PREDS.glob(f'{tag}_part*.npy'): p.unlink()
    return arr

def gt_table(d):
    """Una fila por caja GT (píxeles de la imagen final) con id de anotación 'aid' (empieza en 1)."""
    rows = []
    for r in d.itertuples():
        for (c, x, y, w, h) in r.boxes:
            rows.append(dict(uid=r.uid, split=r.split, location=r.location, cls=c, x=x, y=y, w=w, h=h,
                             W=r.W, H=r.H, rel_area=w * h / (r.W * r.H)))
    g = pd.DataFrame(rows); g.insert(0, 'aid', np.arange(1, len(g) + 1)); return g

def build_coco_gt(d, g):
    ims = [dict(id=int(r.uid), width=int(r.W), height=int(r.H)) for r in d.itertuples()]
    ans = [dict(id=int(r.aid), image_id=int(r.uid), category_id=int(r.cls) + 1, bbox=[r.x, r.y, r.w, r.h],
                area=float(r.w * r.h), iscrowd=0) for r in g.itertuples()]
    gt = COCO(); gt.dataset = dict(images=ims, annotations=ans, categories=[dict(id=i + 1, name=n) for i, n in enumerate(class_names)])
    with contextlib.redirect_stdout(io.StringIO()): gt.createIndex()
    return gt

def _mean_valid(x):
    x = x[x > -1]; return float(x.mean()) if x.size else float('nan')

def coco_eval(gt, dets, img_ids, ignore_aids=()):
    """AP@[.5:.95] y AP@.5 (macro sobre clases con GT) restringido a img_ids. ignore_aids = cajas a ignorar."""
    img_ids = sorted(set(int(i) for i in img_ids))
    sel = dets[np.isin(dets[:, 0], img_ids)]
    if len(sel) == 0: return dict(AP=0.0, AP50=0.0, ap50_c=np.full(NC, np.nan), ap_c=np.full(NC, np.nan))
    ign = [int(a) for a in ignore_aids]
    for a in ign: gt.anns[a]['iscrowd'] = 1
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(sel)
            E = COCOeval(gt, dt, 'bbox'); E.params.imgIds = img_ids
            E.params.areaRng = [[0, 1e10]]; E.params.areaRngLbl = ['all']
            E.evaluate(); E.accumulate()
    finally:
        for a in ign: gt.anns[a]['iscrowd'] = 0
    P = E.eval['precision'][:, :, :, 0, 2]              # [IoU, recall, clase] con maxDets=100
    return dict(AP=_mean_valid(P), AP50=_mean_valid(P[0]),
                ap_c=np.array([_mean_valid(P[:, :, k]) for k in range(P.shape[2])]),
                ap50_c=np.array([_mean_valid(P[0, :, k]) for k in range(P.shape[2])]))

def presence_metrics(d, dets, conf=None):
    """Nivel imagen: ¿hay fauna? Devuelve tasa de FP en vacías, recall y precisión de 'imagen con fauna'."""
    conf = conf or CFG.conf_op
    na = [cls_id[n] + 1 for n in CFG.non_animal if n in cls_id]
    k = dets[(dets[:, 5] >= conf) & ~np.isin(dets[:, 6], na)]
    pred = set(k[:, 0].astype(int))
    is_fauna = d.boxes.apply(lambda bl: any((b[0] + 1) not in na for b in bl))
    is_empty = d.status == 'empty'
    uids = d.uid.astype(int)
    p = uids.isin(pred)
    return dict(n_vacias=int(is_empty.sum()), FP_en_vacias=float(p[is_empty].mean()) if is_empty.any() else np.nan,
                recall_fauna=float(p[is_fauna].mean()) if is_fauna.any() else np.nan,
                precision_fauna=float(is_fauna[p].mean()) if p.any() else np.nan)


# ----------------------------------------------------------------------
# [pred_run]
# ----------------------------------------------------------------------
TEST = ds.df[ds.df.split.isin(['cis_test', 'trans_test'])].reset_index(drop=True)
DETS = {s: predict_split(BEST, ds.df[ds.df.split == s], f'E0_{s}') for s in ['cis_test', 'trans_test']}
DETS_ALL = np.vstack(list(DETS.values()))
GT_TAB = gt_table(TEST)                         # todas las cajas de test (para secciones 7 y 8)
GT = build_coco_gt(TEST, GT_TAB)
print({k: len(v) for k, v in DETS.items()}, 'detecciones guardadas | cajas GT de test:', len(GT_TAB))


# ----------------------------------------------------------------------
# [gap]
# ----------------------------------------------------------------------
rows = []
for kind in ['val', 'test']:
    c, t = G[f'cis_{kind}'], G[f'trans_{kind}']
    for m in ['mAP50', 'mAP50_95']:
        rows.append(dict(partición=kind, métrica=m, cis_ubic_vistas=c[m], trans_ubic_nuevas=t[m],
                         brecha_abs=c[m] - t[m], brecha_rel_pct=100 * (c[m] - t[m]) / c[m] if c[m] else np.nan))
gap = pd.DataFrame(rows).round(4); gap.to_csv(TABLES / '07_brecha_cis_trans.csv'); display(gap)

# control cruzado con pycocotools
chk = {}
for s in ['cis_test', 'trans_test']:
    e = coco_eval(GT, DETS[s], ds.df[ds.df.split == s].uid)
    chk[s] = dict(pycoco_AP50=round(e['AP50'], 4), ultra_AP50=round(G[s]['mAP50'], 4),
                  pycoco_AP50_95=round(e['AP'], 4), ultra_AP50_95=round(G[s]['mAP50_95'], 4))
display(pd.DataFrame(chk).T)

# Por especie: ¿qué especies sufren más al cambiar de ubicación?
sp_gap = sp[['n_train_cajas', 'n_test_cis', 'n_test_trans', 'AP50_95_cis', 'AP50_95_trans']].copy()
sp_gap['caida_abs'] = sp_gap.AP50_95_cis - sp_gap.AP50_95_trans
sp_gap.sort_values('caida_abs', ascending=False).round(3).to_csv(TABLES / '08_brecha_por_especie.csv')
display(sp_gap.sort_values('caida_abs', ascending=False).round(3))


# ----------------------------------------------------------------------
# [per_loc]
# ----------------------------------------------------------------------
from scipy.stats import mannwhitneyu
rows = []
for s in ['cis_test', 'trans_test']:
    dd = ds.df[ds.df.split == s]
    for loc, dl in dd.groupby('location'):
        e = coco_eval(GT, DETS[s], dl.uid)
        rows.append(dict(partición=s, ubicación=loc, n_imgs=len(dl), n_cajas=int(sum(len(b) for b in dl.boxes)),
                         AP50=e['AP50'], AP50_95=e['AP'], **presence_metrics(dl, DETS[s])))
loc_tab = pd.DataFrame(rows); loc_tab.round(3).to_csv(TABLES / '09_ap_por_ubicacion.csv'); display(loc_tab.round(3))

a = loc_tab[(loc_tab.partición == 'cis_test') & (loc_tab.n_cajas >= 20)].AP50_95.dropna()
b = loc_tab[(loc_tab.partición == 'trans_test') & (loc_tab.n_cajas >= 20)].AP50_95.dropna()
if len(a) > 2 and len(b) > 2:
    u = mannwhitneyu(a, b, alternative='greater')
    print(f'AP50-95 por cámara — cis: mediana {a.median():.3f} (n={len(a)}) | trans: mediana {b.median():.3f} (n={len(b)}) | Mann-Whitney (cis > trans): p = {u.pvalue:.4f}')

fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
o = loc_tab.sort_values(['partición', 'AP50_95'])
ax[0].bar(range(len(o)), o.AP50_95, color=np.where(o.partición == 'cis_test', '#3b7dd8', '#e6550d'))
ax[0].set_xticks(range(len(o))); ax[0].set_xticklabels(o.ubicación, rotation=90, fontsize=7)
ax[0].set_title('AP@0.5:0.95 por cámara (azul = vista en train, naranja = nueva)'); ax[0].set_ylabel('AP')
pm = pd.DataFrame({s: presence_metrics(ds.df[ds.df.split == s], DETS[s]) for s in ['cis_test', 'trans_test']}).T
pm[['FP_en_vacias', 'recall_fauna', 'precision_fauna']].plot.bar(ax=ax[1], rot=0)
ax[1].set_title(f'Nivel imagen @conf≥{CFG.conf_op}: falsas alarmas en vacías, recall y precisión de "hay fauna"')
plt.tight_layout(); plt.savefig(FIGS / '05_generalizacion_por_ubicacion.png', dpi=130); plt.close('all')
pm.round(3).to_csv(TABLES / '10_presencia_imagen_cis_vs_trans.csv'); display(pm.round(3))

