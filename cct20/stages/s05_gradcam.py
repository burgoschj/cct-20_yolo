# Etapa s05_gradcam: Grad-CAM y prueba del animal enmascarado
# Se ejecuta con exec() en un espacio de nombres compartido (ver cct20/runner.py).

# ----------------------------------------------------------------------
# [gradcam]
# ----------------------------------------------------------------------
class YoloGradCAM:
    def __init__(self, weights, layers, imgsz):
        self.net = YOLO(str(weights)).model.to(TDEV).float().eval()      # instancia nueva (sin fusionar)
        for p in self.net.parameters(): p.requires_grad_(False)
        self.imgsz, self.acts = imgsz, {}
        for L in layers:                                                  # una capa del backbone por escala P3, P4, P5
            self.net.model[L].register_forward_hook(lambda m, i, o, L=L: self.acts.__setitem__(L, o))
        self.layers = list(layers)

    def __call__(self, img_bgr, target_cls=None):
        h0, w0 = img_bgr.shape[:2]; S = self.imgsz
        r = S / max(h0, w0); nh, nw = round(h0 * r), round(w0 * r)
        top, left = (S - nh) // 2, (S - nw) // 2
        canvas = np.full((S, S, 3), 114, np.uint8); canvas[top:top + nh, left:left + nw] = cv2.resize(img_bgr, (nw, nh))
        x = torch.from_numpy(canvas[..., ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(TDEV).requires_grad_(True)
        with torch.enable_grad():
            out = self.net(x)
            pred = (out[0] if isinstance(out, (list, tuple)) else out)[0]      # (4+nc, n_anclas)
            sc = pred[4:]                                                      # puntuaciones por clase
            if target_cls is None:
                a = int(sc.max(0).values.argmax()); c = int(sc[:, a].argmax())
            else:
                c = int(target_cls); a = int(sc[c].argmax())
            score = sc[c, a]
            n3, n4 = (S // 8) ** 2, (S // 16) ** 2                             # nº de anclas por escala
            level = 0 if a < n3 else (1 if a < n3 + n4 else 2)                 # ¿en qué escala (P3/P4/P5) está la mejor ancla?
            act = self.acts[self.layers[level]]
            grad = torch.autograd.grad(score, act)[0]                          # d puntuación / d activación
        wts = grad.mean((2, 3), keepdim=True)                                  # peso por canal
        cam = torch.relu((wts * act).sum(1))[0].detach().cpu().numpy()         # mapa de calor
        cam = cv2.resize(cam, (S, S))[top:top + nh, left:left + nw]
        cam = cv2.resize(cam, (w0, h0))
        return cam, dict(cls=c, score=float(score.detach()), escala=f'P{level + 3}')

def cam_metrics(cam, box):
    x, y, w, h = [int(round(v)) for v in box]; H, W = cam.shape
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    tot = cam.sum()
    if tot <= 0: return dict(energia_en_caja=np.nan, enriquecimiento=np.nan, pointing=np.nan, area_frac=np.nan)
    e = cam[y1:y2, x1:x2].sum() / tot; af = (x2 - x1) * (y2 - y1) / (W * H)
    py, px = np.unravel_index(cam.argmax(), cam.shape)
    return dict(energia_en_caja=e, enriquecimiento=e / af, pointing=float(x1 <= px < x2 and y1 <= py < y2), area_frac=af)

def detected(uid, boxrow, dets, conf=None):
    """¿Hay una detección de la clase correcta con IoU>=0.5 y conf>=umbral operativo?"""
    conf = conf or CFG.conf_op
    k = dets[(dets[:, 0] == uid) & (dets[:, 5] >= conf) & (dets[:, 6] == boxrow.cls + 1)]
    for _, x, y, w, h, *_ in k:
        iw = max(0, min(x + w, boxrow.x + boxrow.w) - max(x, boxrow.x)); ih = max(0, min(y + h, boxrow.y + boxrow.h) - max(y, boxrow.y))
        inter = iw * ih
        if inter / (w * h + boxrow.w * boxrow.h - inter + 1e-9) >= 0.5: return True
    return False

# --- Muestra: imágenes con UNA sola caja y animal no gigante (la métrica es limpia) ---
N_CAM = CFG.n_cam
cam_rows, cam_store = [], {}
gcam = YoloGradCAM(BEST, CFG.gradcam_layers, CFG.imgsz)
for s in ['cis_test', 'trans_test']:
    cand = GT_TAB[(GT_TAB.split == s) & (GT_TAB.rel_area < 0.5)]
    cand = cand[cand.uid.map(cand.uid.value_counts()) == 1]                       # una caja por imagen
    cand = cand.sample(min(N_CAM, len(cand)), random_state=SEED)
    for r in tqdm(cand.itertuples(), total=len(cand), desc=f'Grad-CAM {s}'):
        path = ds.df.loc[ds.df.uid == r.uid, 'ypath'].iloc[0]
        img = cv2.imread(path); cam, info = gcam(img, target_cls=r.cls)
        m = cam_metrics(cam, (r.x, r.y, r.w, r.h))
        cam_rows.append(dict(split=s, uid=r.uid, location=r.location, cls=class_names[r.cls], escala=info['escala'], detectado=detected(r.uid, r, DETS[s]), **m))
        if len(cam_store) < 400: cam_store[r.uid] = (path, cam, (r.x, r.y, r.w, r.h))
cam_df = pd.DataFrame(cam_rows); cam_df.round(4).to_csv(TABLES / '11_gradcam_metricas.csv', index=False)

resumen_cam = (cam_df.groupby('split').agg(n=('uid', 'size'), energia_en_caja=('energia_en_caja', 'mean'),
               area_frac_media=('area_frac', 'mean'), enriquecimiento_mediana=('enriquecimiento', 'median'),
               pointing=('pointing', 'mean'), tasa_detectado=('detectado', 'mean')).round(3))
display(resumen_cam)
display(cam_df.groupby(['split', 'detectado']).agg(n=('uid', 'size'), enriquecimiento_mediana=('enriquecimiento', 'median'),
                                                    pointing=('pointing', 'mean')).round(3))
resumen_cam.to_csv(TABLES / '12_gradcam_resumen.csv')


# ----------------------------------------------------------------------
# [gradcam_viz]
# ----------------------------------------------------------------------
def overlay(img_bgr, cam, box=None):
    c = cam / (cam.max() + 1e-8)
    heat = cv2.applyColorMap((255 * c).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_bgr, 0.55, heat, 0.45, 0)
    if box is not None:
        x, y, w, h = [int(v) for v in box]; cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for row, s in enumerate(['cis_test', 'trans_test']):
    ids = cam_df[cam_df.split == s].uid.tolist()
    ids = [u for u in ids if u in cam_store][:4]
    for ax, u in zip(axes[row], ids):
        path, cam, box = cam_store[u]; img = cv2.imread(path)
        mt = cam_df[cam_df.uid == u].iloc[0]
        ax.imshow(overlay(img, cam, box)); ax.axis('off')
        ax.set_title(f'{s} · {mt.cls}\nenriquec.={mt.enriquecimiento:.2f} · {"detectado" if mt.detectado else "NO detectado"}', fontsize=10)
plt.suptitle('Grad-CAM (backbone). Verde = caja real del animal. Calor = zonas que más apoyan la clase correcta', y=1.0)
plt.tight_layout(); plt.savefig(FIGS / '06_gradcam_galeria.png', dpi=120); plt.close('all')

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
cam_df.boxplot(column='enriquecimiento', by='split', ax=ax[0]); ax[0].axhline(1, color='r', ls='--'); ax[0].set_ylabel('enriquecimiento (1 = sin foco)')
ax[0].set_title('Enriquecimiento del Grad-CAM en la caja'); plt.suptitle('')
cam_df.boxplot(column='energia_en_caja', by='split', ax=ax[1]); ax[1].set_title('Fracción de energía dentro de la caja'); plt.suptitle('')
plt.tight_layout(); plt.savefig(FIGS / '07_gradcam_metricas.png', dpi=120); plt.close('all')


# ----------------------------------------------------------------------
# [masked]
# ----------------------------------------------------------------------
def run_masked_test(split, n=CFG.n_masked):
    cand = GT_TAB[(GT_TAB.split == split)]
    cand = cand[cand.uid.map(cand.uid.value_counts()) == 1].sample(frac=1, random_state=SEED).head(n)
    empt = ds.df[(ds.df.split == split) & (ds.df.status == 'empty')].sample(frac=1, random_state=SEED)
    model = YOLO(str(BEST)); out = []
    def predict_batch(imgs):
        res = model.predict(source=imgs, conf=CFG.conf_op, iou=CFG.iou_nms, imgsz=CFG.imgsz, **HALF_KW, device=DEVICE, verbose=False)
        return [set(int(c) for c in r.boxes.cls.cpu().numpy()) if r.boxes is not None else set() for r in res]
    imgsA, clsA, imgsB = [], [], []
    for r, e in zip(cand.itertuples(), empt.itertuples()):
        x1, y1, x2, y2 = int(r.x), int(r.y), int(r.x + r.w), int(r.y + r.h)
        a = cv2.imread(ds.df.loc[ds.df.uid == r.uid, 'ypath'].iloc[0]); a[y1:y2, x1:x2] = 114          # tapar animal
        b = cv2.imread(e.ypath); b = cv2.resize(b, (a.shape[1], a.shape[0])); b[y1:y2, x1:x2] = 114   # parche en vacía
        imgsA.append(a); clsA.append(r.cls); imgsB.append(b)
    pa, pb = [], []
    for i in range(0, len(imgsA), 32):
        pa += predict_batch(imgsA[i:i + 32]); pb += predict_batch(imgsB[i:i + 32])
    return dict(partición=split, n=len(imgsA),
                A_cualquier_deteccion=np.mean([len(p) > 0 for p in pa]), A_misma_especie=np.mean([c in p for c, p in zip(clsA, pa)]),
                B_cualquier_deteccion=np.mean([len(p) > 0 for p in pb]))

mk = pd.DataFrame([run_masked_test(s) for s in ['cis_test', 'trans_test']]).round(3)
mk.to_csv(TABLES / '13_prueba_animal_enmascarado.csv', index=False); display(mk)

