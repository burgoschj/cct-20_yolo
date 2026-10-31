# Etapa s06_robustness: Robustez: oclusión y tamaño
# Se ejecuta con exec() en un espacio de nombres compartido (ver cct20/runner.py).

# ----------------------------------------------------------------------
# [occ_feats]
# ----------------------------------------------------------------------
UPATH = dict(zip(ds.df.uid, ds.df.ypath))
rows = []
for uid, g in tqdm(GT_TAB.groupby('uid'), total=GT_TAB.uid.nunique(), desc='características por caja'):
    gray = cv2.imread(UPATH[uid], cv2.IMREAD_GRAYSCALE); H, W = gray.shape
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)); base = lap.mean() + 1e-6
    gl = list(g.itertuples())
    for r in gl:
        x1, y1, x2, y2 = int(r.x), int(r.y), int(r.x + r.w), int(r.y + r.h)
        clutter_raw = float(lap[y1:y2, x1:x2].mean() / base)
        trunc = min(1.0, ((x1 <= 2) + (y1 <= 2) + (x2 >= W - 2) + (y2 >= H - 2)) / 2)
        ov = 0.0
        for o in gl:
            if o.aid == r.aid: continue
            iw = max(0, min(x2, o.x + o.w) - max(x1, o.x)); ih = max(0, min(y2, o.y + o.h) - max(y1, o.y))
            ov = max(ov, iw * ih / (r.w * r.h))
        rows.append((r.aid, clutter_raw, trunc, min(1.0, ov)))
feat = pd.DataFrame(rows, columns=['aid', 'clutter_raw', 'trunc', 'overlap'])
GT_TAB = GT_TAB.drop(columns=[c for c in ['clutter_raw', 'trunc', 'overlap', 'clutter', 'occ_score', 'occ_nivel', 'tam_grupo'] if c in GT_TAB]).merge(feat, on='aid')

GT_TAB['clutter'] = GT_TAB.clutter_raw.rank(pct=True)
GT_TAB['occ_score'] = 0.35 * GT_TAB.trunc + 0.30 * GT_TAB.overlap + 0.35 * GT_TAB.clutter
GT_TAB['occ_nivel'] = pd.qcut(GT_TAB.occ_score.rank(method='first'), 3, labels=['bajo', 'medio', 'alto']).astype(str)

SIZE_BINS = [0, 0.015, 0.05, 1.01]       # según el EDA de CCT-20: ≈25 % / 50 % / 25 % de las cajas
GT_TAB['tam_grupo'] = pd.cut(GT_TAB.rel_area, SIZE_BINS, labels=['pequeño', 'mediano', 'grande'], right=False).astype(str)
GT_TAB['borde'] = np.where(GT_TAB.trunc > 0, 'toca_borde', 'sin_borde')
GT_TAB['solape'] = np.where(GT_TAB.overlap > 0.3, 'solapado>30%', 'sin_solape')

print('Cajas por grupo de oclusión:', GT_TAB.occ_nivel.value_counts().to_dict())
print('Cajas por tamaño:', GT_TAB.tam_grupo.value_counts().to_dict())
print('Tocan el borde:', int((GT_TAB.trunc > 0).sum()), '| solapadas >30 %:', int((GT_TAB.overlap > 0.3).sum()))
GT_TAB.to_csv(TABLES / '14_cajas_test_con_oclusion_y_tamano.csv', index=False)

# Ejemplos visuales de cada nivel (sanidad del proxy)
fig, axes = plt.subplots(3, 6, figsize=(20, 9))
for row, lv in enumerate(['bajo', 'medio', 'alto']):
    sm = GT_TAB[(GT_TAB.occ_nivel == lv) & (GT_TAB.rel_area > 0.01)].sample(6, random_state=SEED)
    for ax, r in zip(axes[row], sm.itertuples()):
        im = cv2.cvtColor(cv2.imread(UPATH[r.uid]), cv2.COLOR_BGR2RGB)
        cv2.rectangle(im, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), (255, 40, 40), 3)
        ax.imshow(im); ax.axis('off'); ax.set_title(f'oclusión {lv} · {class_names[r.cls]}', fontsize=9)
plt.tight_layout(); plt.savefig(FIGS / '08_ejemplos_oclusion.png', dpi=110); plt.close('all')


# ----------------------------------------------------------------------
# [group_eval]
# ----------------------------------------------------------------------
_o = np.argsort(DETS_ALL[:, 0], kind='stable'); _S = DETS_ALL[_o]
_u, _i = np.unique(_S[:, 0], return_index=True)
DET_BY_UID = {int(u): a for u, a in zip(_u, np.split(_S, _i[1:]))}      # detecciones indexadas por imagen

def box_recall(sub, conf=None):
    """Fracción de cajas GT con ≥1 detección de su clase, conf≥umbral e IoU≥0.5."""
    conf = conf or CFG.conf_op; hit = 0
    for r in sub.itertuples():
        a = DET_BY_UID.get(int(r.uid))
        if a is None: continue
        a = a[(a[:, 5] >= conf) & (a[:, 6] == r.cls + 1)]
        if len(a) == 0: continue
        iw = np.clip(np.minimum(a[:, 1] + a[:, 3], r.x + r.w) - np.maximum(a[:, 1], r.x), 0, None)
        ih = np.clip(np.minimum(a[:, 2] + a[:, 4], r.y + r.h) - np.maximum(a[:, 2], r.y), 0, None)
        inter = iw * ih; iou = inter / (a[:, 3] * a[:, 4] + r.w * r.h - inter + 1e-9)
        hit += bool((iou >= 0.5).any())
    return hit / max(1, len(sub))

def eval_by_group(col, order, min_boxes=CFG.min_boxes):
    rows = []
    for gname in order:
        for scope in ['test_todo', 'cis_test', 'trans_test']:
            sub = GT_TAB[GT_TAB[col] == gname]
            if scope != 'test_todo': sub = sub[sub.split == scope]
            if len(sub) < min_boxes:
                rows.append(dict(factor=col, grupo=gname, ámbito=scope, n_cajas=len(sub), n_imgs=np.nan, AP50=np.nan, AP50_95=np.nan, recall_cajas=np.nan, nota='n<%d: no evaluado' % min_boxes)); continue
            img_ids = sub.uid.unique()
            ign = GT_TAB.loc[(~GT_TAB.aid.isin(sub.aid)) & GT_TAB.uid.isin(img_ids), 'aid']      # otras cajas de esas imágenes: se ignoran
            e = coco_eval(GT, DETS_ALL, img_ids, ign)
            rows.append(dict(factor=col, grupo=gname, ámbito=scope, n_cajas=len(sub), n_imgs=len(img_ids),
                             AP50=e['AP50'], AP50_95=e['AP'], recall_cajas=box_recall(sub)))
    return pd.DataFrame(rows)

res_occ  = eval_by_group('occ_nivel', ['bajo', 'medio', 'alto'])
res_size = eval_by_group('tam_grupo', ['grande', 'mediano', 'pequeño'])
res_bord = eval_by_group('borde', ['sin_borde', 'toca_borde'])
res_solp = eval_by_group('solape', ['sin_solape', 'solapado>30%'])
ROB = pd.concat([res_occ, res_size, res_bord, res_solp], ignore_index=True)
ROB.round(3).to_csv(TABLES / '15_robustez_por_grupo.csv', index=False); display(ROB.round(3))

def caida(res, mejor, peor, scope='test_todo'):
    a = res[(res.grupo == mejor) & (res.ámbito == scope)].iloc[0]; b = res[(res.grupo == peor) & (res.ámbito == scope)].iloc[0]
    return dict(factor=a.factor, comparación=f'{mejor} → {peor}', ámbito=scope,
                AP50_95_mejor=a.AP50_95, AP50_95_peor=b.AP50_95, caída_abs=a.AP50_95 - b.AP50_95,
                caída_rel_pct=100 * (a.AP50_95 - b.AP50_95) / a.AP50_95, recall_mejor=a.recall_cajas, recall_peor=b.recall_cajas)
cae = []
for scope in ['test_todo', 'cis_test', 'trans_test']:
    for res, m, p in [(res_occ, 'bajo', 'alto'), (res_size, 'grande', 'pequeño'), (res_bord, 'sin_borde', 'toca_borde'), (res_solp, 'sin_solape', 'solapado>30%')]:
        try: cae.append(caida(res, m, p, scope))
        except (IndexError, KeyError, TypeError): pass
cae = pd.DataFrame(cae).round(3); cae.to_csv(TABLES / '16_caidas_de_desempeno.csv', index=False); display(cae)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for a_, (res, order, ttl) in zip(ax, [(res_occ, ['bajo', 'medio', 'alto'], 'Oclusión estimada (proxy)'), (res_size, ['grande', 'mediano', 'pequeño'], 'Tamaño relativo de la caja')]):
    for scope, c in zip(['cis_test', 'trans_test', 'test_todo'], ['#3b7dd8', '#e6550d', '#555555']):
        y = [res[(res.grupo == g) & (res.ámbito == scope)].AP50_95.values for g in order]
        y = [v[0] if len(v) and not pd.isna(v[0]) else np.nan for v in y]
        a_.plot(order, y, marker='o', color=c, label=scope)
    a_.set_title(ttl); a_.set_ylabel('AP@0.5:0.95'); a_.legend()
plt.tight_layout(); plt.savefig(FIGS / '09_robustez.png', dpi=130); plt.close('all')

