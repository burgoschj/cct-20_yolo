# Etapa s01_data: Datos: descarga, EDA, dataset YOLO
# Se ejecuta con exec() en un espacio de nombres compartido (ver cct20/runner.py).

# ----------------------------------------------------------------------
# [download]
# ----------------------------------------------------------------------
IMG_URLS = [
    'https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz',
    'https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz',
]
ANN_URLS = [
    'https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotation_files.zip',
    'https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_annotation_files.zip',
]
IMG_TAR, ANN_ZIP = RAW / 'images.tar.gz', RAW / 'annotations.zip'
IMG_DIR, ANN_DIR = RAW / 'images', RAW / 'annotations'

def _fetch(u, dest):
    """wget -c si existe (reanuda descargas); si no, requests con streaming."""
    try:
        return subprocess.run(['wget', '-c', '-q', '--show-progress', '-O', str(dest), u]).returncode == 0
    except FileNotFoundError:
        import requests
        try:
            with requests.get(u, stream=True, timeout=60) as r, open(dest, 'wb') as f:
                r.raise_for_status()
                for chunk in r.iter_content(1 << 20): f.write(chunk)
            return True
        except Exception as e:
            print('falló', repr(e)[:120]); return False

def download(urls, dest, min_bytes=1_000_000):
    """Descarga con reintento/continuación (wget -c) probando cada espejo."""
    if dest.exists() and dest.stat().st_size > min_bytes:
        print(f'✔ ya existe {dest.name} ({dest.stat().st_size/1e9:.2f} GB)'); return
    for u in urls:
        print('Descargando', u)
        ok = _fetch(u, dest)
        if ok and dest.exists() and dest.stat().st_size > min_bytes:
            return
    raise RuntimeError('No se pudo descargar. Descarga manualmente desde LILA y súbelo como ' + str(dest))

if not any(IMG_DIR.rglob('*.jpg')) if IMG_DIR.exists() else True:
    download(IMG_URLS, IMG_TAR)
    IMG_DIR.mkdir(exist_ok=True)
    print('Extrayendo imágenes (puede tardar unos minutos)...')
    try: subprocess.run(['tar', '-xzf', str(IMG_TAR), '-C', str(IMG_DIR)], check=True)
    except FileNotFoundError:
        with tarfile.open(IMG_TAR) as tf: tf.extractall(IMG_DIR)
    IMG_TAR.unlink()          # libera ~6 GB de disco
if not any(ANN_DIR.rglob('*.json')) if ANN_DIR.exists() else True:
    ANN_DIR.mkdir(exist_ok=True)
    # Copia local opcional (útil si LILA bloquea la descarga): OUT/datos/eccv_18_annotation_files.zip
    _local = OUT / 'datos' / 'eccv_18_annotation_files.zip'
    if _local.exists():
        print('Anotaciones desde copia local:', _local); ANN_ZIP = _local
    else:
        download(ANN_URLS, ANN_ZIP)
    zipfile.ZipFile(ANN_ZIP).extractall(ANN_DIR)

print('Imágenes .jpg:', sum(1 for _ in IMG_DIR.rglob('*.jpg')))
print('Archivos JSON:', [p.name for p in ANN_DIR.rglob('*.json')])


# ----------------------------------------------------------------------
# [load_ann]
# ----------------------------------------------------------------------
SPLITS = ['train', 'cis_val', 'cis_test', 'trans_val', 'trans_test']

def find_ann_files(ann_dir):
    """Busca <split>_*.json sin asumir la subcarpeta."""
    found = {}
    for p in sorted(Path(ann_dir).rglob('*.json')):
        n = p.stem.lower()
        for s in SPLITS:
            if n.startswith(s) and len(n) > len(s) and n[len(s)] == '_' and s not in found:
                found[s] = p
    return found

ann_files = find_ann_files(ANN_DIR)
print({k: v.name for k, v in ann_files.items()})
assert set(ann_files) == set(SPLITS), f'Faltan particiones: {set(SPLITS) - set(ann_files)}. Revisa los nombres impresos arriba.'

# Índice nombre_de_archivo -> ruta real (robusto a subcarpetas)
img_index = {p.name: p for p in IMG_DIR.rglob('*.jpg')}

def load_coco_split(split, jpath):
    d = json.load(open(jpath))
    cats = {c['id']: c['name'] for c in d['categories']}
    by_img = defaultdict(list)
    for a in d['annotations']:
        by_img[a['image_id']].append(a)
    rows, missing = [], 0
    for im in d['images']:
        p = img_index.get(Path(im['file_name']).name)
        if p is None:
            missing += 1; continue
        boxes, has_empty, nobox = [], False, 0
        for a in by_img.get(im['id'], []):
            name = cats[a['category_id']]
            if name == 'empty':
                has_empty = True
            elif a.get('bbox'):
                x, y, w, h = a['bbox']; boxes.append((name, x, y, w, h))
            else:
                nobox += 1
        n_ann = len(by_img.get(im['id'], []))
        # animal = tiene caja. empty = etiquetada 'empty'. unlabeled = etiqueta de ANIMAL pero SIN caja:
        # en CCT-20 hay ~1.4 k así en train; NO son vacías (contienen animales). Se excluyen de entrenamiento y evaluación.
        status = 'animal' if boxes else ('unlabeled' if nobox > 0 else 'empty')
        rows.append(dict(split=split, image_id=im['id'], fname=p.name, raw_path=str(p),
                         Wj=im.get('width'), Hj=im.get('height'),
                         location=str(im.get('location')), seq_id=str(im.get('seq_id')),
                         frame_num=im.get('frame_num'),
                         date=im.get('date_captured') or im.get('datetime'),
                         boxes_json=boxes, status=status, n_ann=n_ann))
    if missing: print(f'⚠️ {split}: {missing} imágenes del JSON no están en disco')
    return pd.DataFrame(rows)

df_all = pd.concat([load_coco_split(s, ann_files[s]) for s in SPLITS], ignore_index=True)
df_all['uid'] = np.arange(1, len(df_all) + 1)          # id único global (empieza en 1: pycocotools lo requiere)

# Tamaño real de cada archivo (por si el JSON hablaba de otra resolución)
sizes = [Image.open(p).size for p in tqdm(df_all.raw_path, desc='leyendo tamaños')]
df_all['W0'] = [s[0] for s in sizes]; df_all['H0'] = [s[1] for s in sizes]
df_all['Wj'] = df_all.Wj.fillna(df_all.W0); df_all['Hj'] = df_all.Hj.fillna(df_all.H0)
print('Escala JSON→archivo (debería ser 1.0):', (df_all.W0 / df_all.Wj).round(3).value_counts().head(3).to_dict())

# Nombres de clase (sin "empty"), índice estable alfabético
class_names = sorted({b[0] for bs in df_all.boxes_json for b in bs})
NC = len(class_names)
cls_id = {n: i for i, n in enumerate(class_names)}
print(f'{NC} clases:', class_names)

# ---------- Chequeos anti-fuga ----------
locs = {s: set(df_all[df_all.split == s].location) for s in SPLITS}
seqs = {s: set(df_all[df_all.split == s].seq_id) for s in SPLITS}
print('\nUbicaciones por partición:', {s: len(v) for s, v in locs.items()})
for s in ['trans_val', 'trans_test']:
    ov = locs[s] & locs['train']
    print(f'  {"✅" if not ov else "⚠️"} {s} ∩ train (ubicaciones) = {len(ov)}')
for s in ['cis_val', 'cis_test']:
    print(f'  ℹ️ {s}: {len(locs[s] & locs["train"])}/{len(locs[s])} ubicaciones también están en train (esperado: todas)')
print(f'  {"✅" if not (locs["trans_val"] & locs["trans_test"]) else "ℹ️"} trans_val ∩ trans_test (ubicaciones) = {len(locs["trans_val"] & locs["trans_test"])}')
for a in SPLITS:
    for b in SPLITS:
        if a < b and (seqs[a] & seqs[b]):
            if {a, b} == {'train', 'cis_val'}:
                print(f'  ℹ️ secuencias compartidas train/cis_val: {len(seqs[a] & seqs[b])} → ESPERADO: en el benchmark oficial cis_val se saca de los mismos días de train. '
                      'Por eso cis_val NO es independiente: no lo uses para elegir el modelo ni como prueba de generalización.')
            else:
                print(f'  ⚠️ secuencias compartidas entre {a} y {b}: {len(seqs[a] & seqs[b])}')
print('  (un ⚠️ de secuencias indicaría fuga real; train/cis_val es la excepción de diseño)')

resumen = (df_all.groupby('split')
           .agg(imagenes=('uid', 'size'), ubicaciones=('location', 'nunique'),
                con_fauna=('status', lambda s: (s == 'animal').sum()),
                vacias=('status', lambda s: (s == 'empty').sum()),
                animal_sin_caja=('status', lambda s: (s == 'unlabeled').sum()))
           .reindex(SPLITS))
resumen['pct_vacias'] = (100 * resumen.vacias / (resumen.con_fauna + resumen.vacias)).round(1)      # % de vacías entre las imágenes UTILIZABLES
resumen.to_csv(TABLES / '01_resumen_particiones.csv')
resumen


# ----------------------------------------------------------------------
# [check_unlabeled]
# ----------------------------------------------------------------------
cand = df_all[df_all.status == 'unlabeled']
print(f'{len(cand)} imágenes con etiqueta de animal pero sin caja (excluidas)')
if len(cand):
    sm = cand.sample(min(12, len(cand)), random_state=SEED)
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    for ax, r in zip(axes.ravel(), sm.itertuples()):
        ax.imshow(cv2.cvtColor(cv2.imread(r.raw_path), cv2.COLOR_BGR2RGB)); ax.axis('off'); ax.set_title(f'{r.split} · loc {r.location}', fontsize=9)
    plt.tight_layout(); plt.close('all')


# ----------------------------------------------------------------------
# [eda]
# ----------------------------------------------------------------------
# Tabla "una fila por caja"
bx = pd.DataFrame([(r.split, r.uid, r.location, b[0], b[3] * b[4] / (r.Wj * r.Hj))
                   for r in df_all.itertuples() for b in r.boxes_json],
                  columns=['split', 'uid', 'location', 'species', 'rel_area'])

sp_tab = bx.pivot_table(index='species', columns='split', values='uid', aggfunc='size', fill_value=0).reindex(columns=SPLITS)
sp_tab['train_imgs'] = bx[bx.split == 'train'].groupby('species').uid.nunique()
sp_tab = sp_tab.fillna(0).astype(int).sort_values('train', ascending=False)
sp_tab.to_csv(TABLES / '02_cajas_por_especie_y_particion.csv')
display(sp_tab)

tr = sp_tab.train[sp_tab.train > 0]
print(f'Razón de desbalance en train (más común / más rara): {tr.max()} / {tr.min()} = {tr.max()/tr.min():.0f}×')

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
tr.sort_values().plot.barh(ax=ax[0], color='#3b7dd8'); ax[0].set_xscale('log')
ax[0].set_title('Cajas por especie en TRAIN (escala log)'); ax[0].set_xlabel('n.º de cajas')
ax[1].hist(np.log10(bx.rel_area), bins=60, color='#3b7dd8')
ax[1].set_title('Tamaño relativo de la caja (log10 área/área imagen)'); ax[1].set_xlabel('log10(área relativa)')
for q in [0.01, 0.05]:
    ax[1].axvline(np.log10(q), color='r', ls='--'); ax[1].text(np.log10(q), ax[1].get_ylim()[1]*0.9, f'{q:.0%}', color='r')
loc_n = df_all.groupby(['split', 'location']).size().reset_index(name='n')
for s, c in zip(SPLITS, ['#3b7dd8', '#7fb3f5', '#9ecae1', '#e6550d', '#fdae6b']):
    sub = loc_n[loc_n.split == s].sort_values('n', ascending=False)
    ax[2].bar([f'{s}:{l}' for l in sub.location], sub.n, color=c, label=s)
ax[2].set_xticks([]); ax[2].set_title('Imágenes por ubicación de cámara (color = partición)'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / '01_eda.png', dpi=130); plt.close('all')

print('Cuantiles del área relativa de las cajas:', bx.rel_area.quantile([.1, .25, .5, .75, .9]).round(4).to_dict())


# ----------------------------------------------------------------------
# [build_ds]
# ----------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor

def _write_image(src, dst, size):
    """Hardlink si no hay que redimensionar (instantáneo, no duplica disco); si no, redimensiona."""
    if dst.exists(): return
    if size is None:
        try: os.link(src, dst)
        except OSError: shutil.copy2(src, dst)
    else:
        im = cv2.imread(str(src)); im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])

def build_dataset(name, max_side):
    root = ROOT / 'cct_yolo' / name
    d = df_all[df_all.status != 'unlabeled'].copy()
    print('Imágenes usadas:', len(d), '| excluidas (sin caja):', (df_all.status == 'unlabeled').sum())

    # tamaño final por imagen
    scale = np.ones(len(d)) if not max_side else np.minimum(1.0, max_side / np.maximum(d.W0, d.H0))
    d['W'] = np.round(d.W0 * scale).astype(int); d['H'] = np.round(d.H0 * scale).astype(int)
    d['resize'] = scale < 1.0

    for s in SPLITS:
        (root / 'images' / s).mkdir(parents=True, exist_ok=True); (root / 'labels' / s).mkdir(parents=True, exist_ok=True)
    d['ypath'] = [str(root / 'images' / s / f) for s, f in zip(d.split, d.fname)]

    # 1) imágenes
    jobs = [(r.raw_path, Path(r.ypath), (r.W, r.H) if r.resize else None) for r in d.itertuples()]
    with ThreadPoolExecutor(4) as ex:
        list(tqdm(ex.map(lambda a: _write_image(*a), jobs), total=len(jobs), desc='imágenes'))

    # 2) etiquetas YOLO + cajas en píxeles finales
    boxes_px, n_drop = [], 0
    for r in tqdm(d.itertuples(), total=len(d), desc='etiquetas'):
        lines, bl = [], []
        for (name_, x, y, w, h) in r.boxes_json:
            x, y, w, h = x * r.W / r.Wj, y * r.H / r.Hj, w * r.W / r.Wj, h * r.H / r.Hj   # a píxeles de la imagen final
            x1, y1, x2, y2 = max(0, x), max(0, y), min(r.W, x + w), min(r.H, y + h)     # recorta a la imagen
            if (x2 - x1) < 2 or (y2 - y1) < 2: n_drop += 1; continue                     # caja degenerada
            bl.append((cls_id[name_], x1, y1, x2 - x1, y2 - y1))
            lines.append(f'{cls_id[name_]} {(x1+x2)/2/r.W:.6f} {(y1+y2)/2/r.H:.6f} {(x2-x1)/r.W:.6f} {(y2-y1)/r.H:.6f}')
        boxes_px.append(bl)
        # imagen vacía => archivo .txt VACÍO (= fondo)
        (root / 'labels' / r.split / (Path(r.fname).stem + '.txt')).write_text('\n'.join(lines))
    d['boxes'] = boxes_px
    # si al recortar se quedó sin cajas, pasa a 'empty'? No: se descarta para no enseñar un animal como fondo
    bad = (d.status == 'animal') & (d.boxes.apply(len) == 0)
    print('Cajas degeneradas descartadas:', n_drop, '| imágenes con animal que quedaron sin cajas (se quitan):', int(bad.sum()))
    d = d[~bad].reset_index(drop=True)

    # 2b) FONDOS: el train oficial no trae imágenes vacías. Se toman las vacías de cis_val (mismas cámaras de train; cis_val no se usa para elegir el modelo ni como test)
    n_bg = 0
    if CFG.bg_from_cis_val:
        for sub_ in ('images', 'labels'): (root / sub_ / 'train_bg').mkdir(parents=True, exist_ok=True)
        for r in d[(d.split == 'cis_val') & (d.status == 'empty')].itertuples():
            di = root / 'images' / 'train_bg' / r.fname
            if not di.exists():
                try: os.link(r.ypath, di)
                except OSError: shutil.copy2(r.ypath, di)
            (root / 'labels' / 'train_bg' / (Path(r.fname).stem + '.txt')).write_text('')      # .txt vacío = fondo
            n_bg += 1
    print(f'Fondos añadidos al entrenamiento (vacías de cis_val): {n_bg}')

    # 3) yaml
    names = {i: n for i, n in enumerate(class_names)}
    def wy(fname, train, val):
        p = root / fname
        yaml.safe_dump(dict(path=str(root), train=train, val=val, names=names), open(p, 'w'), sort_keys=False)
        return p
    vals = [f'images/{s}' for s in CFG.val_splits]
    ymls = dict(train=wy('data.yaml', ['images/train', 'images/train_bg'] if n_bg else 'images/train', vals),
                train_only=wy('data_train_only.yaml', 'images/train', vals))
    for s in ['cis_val', 'trans_val', 'cis_test', 'trans_test']:
        ymls[s] = wy(f'data_{s}.yaml', 'images/train', f'images/{s}')
    ymls['test_all'] = wy('data_test_all.yaml', 'images/train', ['images/cis_test', 'images/trans_test'])
    ds = NS(name=name, root=root, df=d, yaml=ymls, n_bg=n_bg)
    d.drop(columns=['boxes_json']).to_pickle(OUT / f'dataset_{name}.pkl')
    return ds

ds = build_dataset('s%d' % CFG.max_side if CFG.max_side else 'orig', CFG.max_side)
print(ds.df.groupby(['split', 'status']).size().unstack(fill_value=0).reindex(SPLITS))


# ----------------------------------------------------------------------
# [check_labels]
# ----------------------------------------------------------------------
def draw_yolo(img_path, ax, title=''):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
    lab = Path(str(img_path).replace('/images/', '/labels/')).with_suffix('.txt')
    for ln in lab.read_text().splitlines():
        c, cx, cy, w, h = ln.split(); c = int(c); cx, cy, w, h = float(cx)*W, float(cy)*H, float(w)*W, float(h)*H
        cv2.rectangle(img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (255, 40, 40), 3)
        cv2.putText(img, class_names[c], (int(cx-w/2), max(15, int(cy-h/2)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    ax.imshow(img); ax.set_title(title, fontsize=9); ax.axis('off')

samp = ds.df[ds.df.split == 'train'].sample(8, random_state=SEED)
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, r in zip(axes.ravel(), samp.itertuples()):
    draw_yolo(r.ypath, ax, f'{r.status} · loc {r.location}')
plt.tight_layout(); plt.savefig(FIGS / '02_verificacion_etiquetas.png', dpi=110); plt.close('all')

