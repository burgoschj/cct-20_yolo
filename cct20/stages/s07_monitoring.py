# Etapa s07_monitoring: Módulo de monitoreo y demo en cámara no vista
# Se ejecuta con exec() en un espacio de nombres compartido (ver cct20/runner.py).

# ----------------------------------------------------------------------
# [monitor_module]
# ----------------------------------------------------------------------
from dataclasses import dataclass, field

@dataclass
class MonitorConfig:
    conf_default: float = 0.25                           # R1
    conf_by_species: dict = field(default_factory=dict)  # R1: p. ej. {'bobcat': 0.15}
    min_rel_area: float = 0.0005                         # R2
    conf_confirm: float = 0.60                           # R3
    min_frames_confirm: int = 2                          # R3
    burst_gap_s: float = 10.0                            # R3 (si no hay seq_id)
    non_animal: tuple = ('car',)                         # R4
    event_gap_min: float = 30.0                          # R5

def make_frames(d):
    """Tabla de fotogramas de UNA o varias cámaras: uid, cámara, fecha-hora, ráfaga, tamaño."""
    f = pd.DataFrame(dict(uid=d.uid.values, camera=d.location.values, datetime=pd.to_datetime(d.date.values, errors='coerce'),
                          seq=d.seq_id.values, W=d.W.values, H=d.H.values))
    bad = f.datetime.isna().sum()
    if bad: print(f'⚠️ {bad} fotogramas sin fecha válida: se excluyen del reporte temporal')
    return f.dropna(subset=['datetime']).sort_values(['camera', 'datetime']).reset_index(drop=True)

def add_bursts(f, cfg):
    """Ráfaga = seq_id si existe; si no, agrupa por proximidad temporal."""
    f = f.copy()
    if f.seq.notna().all() and (f.seq != 'None').all():
        f['burst'] = f.camera.astype(str) + '|' + f.seq.astype(str)
    else:
        gap = f.groupby('camera').datetime.diff().dt.total_seconds()
        f['burst'] = f.camera.astype(str) + '|' + ((gap.isna() | (gap > cfg.burst_gap_s)).groupby(f.camera).cumsum()).astype(str)
    return f

def apply_rules(dets_arr, frames, cfg):
    """dets_arr: Nx7 [uid,x,y,w,h,conf,clase+1]. Devuelve detecciones CONFIRMADAS con columna is_fauna."""
    f = add_bursts(frames, cfg)
    d = pd.DataFrame(dets_arr, columns=['uid', 'x', 'y', 'w', 'h', 'conf', 'cls']); d['uid'] = d.uid.astype(int); d['cls'] = d.cls.astype(int) - 1
    d['species'] = d.cls.map(dict(enumerate(class_names)))
    d = d.merge(f[['uid', 'camera', 'datetime', 'burst', 'W', 'H']], on='uid', how='inner')
    thr = d.species.map(cfg.conf_by_species).fillna(cfg.conf_default)
    d = d[(d.conf >= thr) & (d.w * d.h / (d.W * d.H) >= cfg.min_rel_area)]                 # R1 + R2
    g = d.groupby(['burst', 'species']).agg(max_conf=('conf', 'max'), n_frames=('uid', 'nunique')).reset_index()
    g['ok'] = (g.max_conf >= cfg.conf_confirm) | (g.n_frames >= cfg.min_frames_confirm)    # R3
    d = d.merge(g[['burst', 'species', 'ok']], on=['burst', 'species']); d = d[d.ok].drop(columns='ok')
    d['is_fauna'] = ~d.species.isin(cfg.non_animal)                                         # R4
    return d.reset_index(drop=True)

def assign_period(dt, mode):
    if mode == 'dia':     return dt.dt.strftime('%Y-%m-%d')
    if mode == 'semana':  return dt.dt.strftime('%G-S%V')
    if mode == 'mes':     return dt.dt.strftime('%Y-%m')
    if mode == 'hora':    return dt.dt.hour.map(lambda h: f'{h:02d}h')
    if mode == 'franja':  return pd.cut(dt.dt.hour, [-1, 5, 11, 17, 23], labels=['1) 00-06 madrugada', '2) 06-12 mañana', '3) 12-18 tarde', '4) 18-24 noche']).astype(str)
    raise ValueError(mode)

def build_reports(frames, dets, cfg, period='franja'):
    fa = dets[dets.is_fauna].reset_index(drop=True)
    f = frames.copy()
    pf = fa.groupby('uid').size().rename('n_individuos')
    f = f.merge(pf, on='uid', how='left'); f['n_individuos'] = f.n_individuos.fillna(0).astype(int); f['con_fauna'] = f.n_individuos > 0
    # R5: eventos independientes
    ev = fa.sort_values(['camera', 'species', 'datetime']).reset_index(drop=True)
    gap = ev.groupby(['camera', 'species']).datetime.diff().dt.total_seconds() / 60
    ev['evento'] = (gap.isna() | (gap > cfg.event_gap_min)).astype(int).groupby([ev.camera, ev.species]).cumsum()
    ev['periodo'] = assign_period(ev.datetime, period)
    # resumen por cámara
    cam = f.groupby('camera').agg(fotogramas=('uid', 'size'), con_fauna=('con_fauna', 'sum'), inicio=('datetime', 'min'), fin=('datetime', 'max'))
    cam['vacias'] = cam.fotogramas - cam.con_fauna
    cam['pct_vacias'] = (100 * cam.vacias / cam.fotogramas).round(1); cam['pct_con_fauna'] = (100 - cam.pct_vacias).round(1)
    cam['especies_distintas'] = fa.groupby('camera').species.nunique(); cam['detecciones'] = fa.groupby('camera').size()
    cam['eventos'] = ev.drop_duplicates(['camera', 'species', 'evento']).groupby('camera').size()
    cam = cam.fillna({'especies_distintas': 0, 'detecciones': 0, 'eventos': 0}).astype({'especies_distintas': int, 'detecciones': int, 'eventos': int})
    # actividad por cámara × especie × periodo
    if len(ev):
        cnt = ev.groupby(['camera', 'species', 'periodo', 'uid']).size()
        act = ev.groupby(['camera', 'species', 'periodo']).agg(fotogramas=('uid', 'nunique'), detecciones=('uid', 'size'),
                                                                eventos=('evento', 'nunique'), conf_media=('conf', 'mean'))
        act['max_individuos_por_foto'] = cnt.groupby(level=[0, 1, 2]).max()
        act = act.round({'conf_media': 3}).reset_index()
    else:
        act = pd.DataFrame(columns=['camera', 'species', 'periodo', 'fotogramas', 'detecciones', 'eventos', 'conf_media', 'max_individuos_por_foto'])
    return dict(camara=cam.reset_index(), actividad=act, fotogramas=f, eventos=ev)

def gt_as_dets(d):
    """Detecciones 'perfectas' construidas con las etiquetas reales (conf=1) para obtener el reporte verdadero."""
    rows = [(r.uid, x, y, w, h, 1.0, c + 1) for r in d.itertuples() for (c, x, y, w, h) in r.boxes]
    return np.array(rows, dtype=float).reshape(-1, 7)


# ----------------------------------------------------------------------
# [calib]
# ----------------------------------------------------------------------
VAL = ds.df[ds.df.split == 'trans_val'].reset_index(drop=True)
DETS_VAL = predict_split(BEST, VAL, 'E0_trans_val')
frames_val = make_frames(VAL)

def presence_set(dets_df, uids):
    s = dets_df[dets_df.is_fauna]; return set(zip(s.uid.astype(int), s.species))
gt_val = apply_rules(gt_as_dets(VAL), frames_val, MonitorConfig(conf_default=0.0, conf_confirm=0.0, min_rel_area=0.0))
truth = presence_set(gt_val, frames_val.uid)

rows = []
for cd in [0.15, 0.25, 0.35, 0.50]:
    for cc in [0.50, 0.60, 0.80]:
        for mf in [1, 2]:
            cfg = MonitorConfig(conf_default=cd, conf_confirm=cc, min_frames_confirm=mf)
            pred = presence_set(apply_rules(DETS_VAL, frames_val, cfg), frames_val.uid)
            tp = len(pred & truth); p = tp / max(1, len(pred)); r = tp / max(1, len(truth))
            rows.append(dict(conf_default=cd, conf_confirm=cc, min_frames=mf, precision=p, recall=r, F1=2 * p * r / max(1e-9, p + r)))
grid = pd.DataFrame(rows).sort_values('F1', ascending=False).round(3); grid.to_csv(TABLES / '18_calibracion_umbrales_val.csv', index=False)
display(grid.head(8))
best = grid.iloc[0]
CFG_MON = MonitorConfig(conf_default=float(best.conf_default), conf_confirm=float(best.conf_confirm), min_frames_confirm=int(best.min_frames))
print('Configuración elegida con VALIDACIÓN:', CFG_MON)


# ----------------------------------------------------------------------
# [demo]
# ----------------------------------------------------------------------
cand = ds.df[ds.df.split == 'trans_test'].groupby('location').size().sort_values(ascending=False)
DEMO_LOC = cand.index[0]
d_demo = ds.df[(ds.df.split == 'trans_test') & (ds.df.location == DEMO_LOC)].reset_index(drop=True)
print(f'Cámara de demostración: {DEMO_LOC} ({len(d_demo)} fotogramas) — NO aparece en entrenamiento: {DEMO_LOC not in set(ds.df[ds.df.split=="train"].location)}')

frames = make_frames(d_demo)
dets_demo = DETS['trans_test'][np.isin(DETS['trans_test'][:, 0], d_demo.uid.values)]     # predicciones del modelo sobre esa cámara
conf_dets = apply_rules(dets_demo, frames, CFG_MON)
REP = build_reports(frames, conf_dets, CFG_MON, period='franja')

print('\n=== Resumen por cámara ==='); display(REP['camara'])
print('\n=== Actividad por especie y periodo (franja del día) ==='); display(REP['actividad'])

# Periodo diario y por hora, para gráficos
REP_D = build_reports(frames, conf_dets, CFG_MON, period='dia'); REP_H = build_reports(frames, conf_dets, CFG_MON, period='hora')
with pd.ExcelWriter(OUT / 'reporte_monitoreo.xlsx', engine='openpyxl') as xw:
    REP['camara'].to_excel(xw, sheet_name='resumen_camara', index=False)
    REP['actividad'].to_excel(xw, sheet_name='actividad_franja', index=False)
    REP_H['actividad'].to_excel(xw, sheet_name='actividad_hora', index=False)
    REP_D['actividad'].to_excel(xw, sheet_name='actividad_dia', index=False)
    REP['eventos'].to_excel(xw, sheet_name='detecciones_confirmadas', index=False)
REP['camara'].to_csv(TABLES / '19_monitoreo_resumen_camara.csv', index=False)
REP['actividad'].to_csv(TABLES / '20_monitoreo_actividad_franja.csv', index=False)
REP_H['actividad'].to_csv(TABLES / '21_monitoreo_actividad_hora.csv', index=False)
print('Guardado:', OUT / 'reporte_monitoreo.xlsx')


# ----------------------------------------------------------------------
# [demo_plots]
# ----------------------------------------------------------------------
# Mapa de calor especie × hora y línea de tiempo diaria
h = REP_H['actividad'].pivot_table(index='species', columns='periodo', values='detecciones', aggfunc='sum', fill_value=0)
h = h.reindex(columns=[f'{i:02d}h' for i in range(24)], fill_value=0)
dd = REP_D['fotogramas'].assign(dia=lambda x: x.datetime.dt.date).groupby('dia').agg(fotos=('uid', 'size'), con_fauna=('con_fauna', 'sum'))
fig, ax = plt.subplots(1, 2, figsize=(17, 4.5), gridspec_kw={'width_ratios': [1.3, 1]})
if len(h):
    im = ax[0].imshow(h.values, aspect='auto', cmap='YlOrRd'); ax[0].set_yticks(range(len(h))); ax[0].set_yticklabels(h.index)
    ax[0].set_xticks(range(24)); ax[0].set_xticklabels(range(24)); ax[0].set_xlabel('hora del día'); plt.colorbar(im, ax=ax[0], label='detecciones')
ax[0].set_title(f'Cámara {DEMO_LOC}: actividad por especie y hora (reporte automático)')
ax[1].plot(dd.index, dd.fotos, label='fotogramas'); ax[1].plot(dd.index, dd.con_fauna, label='con fauna'); ax[1].legend()
ax[1].set_title('Fotogramas por día'); plt.setp(ax[1].get_xticklabels(), rotation=45)
plt.tight_layout(); plt.savefig(FIGS / '10_monitoreo_demo.png', dpi=130); plt.close('all')


# ----------------------------------------------------------------------
# [demo_valid]
# ----------------------------------------------------------------------
gt_conf = apply_rules(gt_as_dets(d_demo), frames, MonitorConfig(conf_default=0.0, conf_confirm=0.0, min_rel_area=0.0, non_animal=CFG.non_animal))
REP_GT = build_reports(frames, gt_conf, CFG_MON, period='franja')

m = REP['actividad'].merge(REP_GT['actividad'], on=['camera', 'species', 'periodo'], how='outer', suffixes=('_modelo', '_real')).fillna(0)
m['err_detecciones'] = (m.detecciones_modelo - m.detecciones_real).abs(); m['err_eventos'] = (m.eventos_modelo - m.eventos_real).abs()
print(f'MAE detecciones por (especie, franja): {m.err_detecciones.mean():.2f} | MAE eventos: {m.err_eventos.mean():.2f}')
tot = m.groupby('species')[['detecciones_modelo', 'detecciones_real', 'eventos_modelo', 'eventos_real']].sum()
tot['err_rel_detecciones_%'] = (100 * (tot.detecciones_modelo - tot.detecciones_real) / tot.detecciones_real.replace(0, np.nan)).round(1)
display(tot.sort_values('detecciones_real', ascending=False))
tot.to_csv(TABLES / '22_monitoreo_modelo_vs_real_por_especie.csv')

pc = REP['camara'].merge(REP_GT['camara'], on='camera', suffixes=('_modelo', '_real'))
display(pc[['camera', 'pct_vacias_modelo', 'pct_vacias_real', 'especies_distintas_modelo', 'especies_distintas_real', 'eventos_modelo', 'eventos_real']])

pred_set = set(zip(conf_dets[conf_dets.is_fauna].uid.astype(int), conf_dets[conf_dets.is_fauna].species))
tru_set = set(zip(gt_conf[gt_conf.is_fauna].uid.astype(int), gt_conf[gt_conf.is_fauna].species))
tp = len(pred_set & tru_set); p = tp / max(1, len(pred_set)); r = tp / max(1, len(tru_set))
print(f'Presencia de especie por fotograma → precisión {p:.3f} · recall {r:.3f} · F1 {2*p*r/max(1e-9,p+r):.3f}')


# ----------------------------------------------------------------------
# [camara_confusion]
# ----------------------------------------------------------------------
# ---- Matriz de confusión de las cajas reales en la cámara de demostración (umbral operativo, IoU>=0.5) ----
def _iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih; u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0

_det = dets_demo[dets_demo[:, 5] >= CFG.conf_op]
_by = {u: _det[_det[:, 0] == u] for u in np.unique(_det[:, 0])}
_rows = []
for r in GT_TAB[GT_TAB.uid.isin(d_demo.uid)].itertuples():
    pred, best = '(no detectada)', 0.5
    for x in _by.get(r.uid, np.zeros((0, 7))):
        v = _iou_xywh((r.x, r.y, r.w, r.h), x[1:5])
        if v >= best: best, pred = v, class_names[int(x[6]) - 1]
    _rows.append(dict(real=class_names[int(r.cls)], pred=pred))
if _rows:
    _cm = pd.crosstab(pd.DataFrame(_rows).real, pd.DataFrame(_rows).pred)
    _cm.to_csv(TABLES / '19_confusion_camara_demo.csv'); display(_cm)

