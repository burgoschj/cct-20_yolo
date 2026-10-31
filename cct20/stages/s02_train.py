# Etapa s02_train: Entrenamiento YOLOv8s (E0)
# Se ejecuta con exec() en un espacio de nombres compartido (ver cct20/runner.py).

# ----------------------------------------------------------------------
# [train_fn]
# ----------------------------------------------------------------------
from ultralytics import YOLO

def train_experiment(name, data_yaml, imgsz=None, model_name=None, **over):
    run_dir = RUNS / name; run_dir.mkdir(parents=True, exist_ok=True)
    last, best, done = run_dir / 'weights' / 'last.pt', run_dir / 'weights' / 'best.pt', run_dir / 'FIN.txt'
    if done.exists():
        print(f'✔ {name} ya entrenado → {best}'); return best

    args = dict(data=str(data_yaml), imgsz=imgsz or CFG.imgsz, epochs=CFG.epochs, batch=CFG.batch,
                workers=CFG.workers, patience=CFG.patience, device=DEVICE, project=str(RUNS), name=name, exist_ok=True,
                seed=SEED, pretrained=True, amp=True, cache=False, plots=CFG.train_plots,
                # optimización
                optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=5e-4, warmup_epochs=3, cos_lr=True,
                # aumento de datos
                mosaic=1.0, close_mosaic=10, mixup=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5)
    args.update(over)
    json.dump({'model': model_name or CFG.model, **args}, open(run_dir / 'decisiones_entrenamiento.json', 'w'), indent=2)

    if last.exists():                       # ¿se cayó la sesión? → reanudar
        print('↻ Reanudando desde', last)
        try:
            YOLO(str(last)).train(resume=True)
        except Exception as e:
            print('No se pudo reanudar (¿ya estaba completo?):', repr(e)[:200])
    else:
        YOLO(model_name or CFG.model).train(**args)
    done.write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
    return best


# ----------------------------------------------------------------------
# [train_e0]
# ----------------------------------------------------------------------
# ⏱ Mide la duración de la 1.ª época en el log y extrapola al total.
BEST = train_experiment('E0_base', ds.yaml['train'])
print('Pesos:', BEST)


# ----------------------------------------------------------------------
# [curves]
# ----------------------------------------------------------------------
res = pd.read_csv(RUNS / 'E0_base' / 'results.csv'); res.columns = [c.strip() for c in res.columns]
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ax[0].plot(res.epoch, res['train/box_loss'], label='train box'); ax[0].plot(res.epoch, res['val/box_loss'], label='val box')
ax[0].plot(res.epoch, res['train/cls_loss'], '--', label='train cls'); ax[0].plot(res.epoch, res['val/cls_loss'], '--', label='val cls')
ax[0].set_title('Pérdidas'); ax[0].legend()
ax[1].plot(res.epoch, res['metrics/mAP50(B)'], label='mAP50'); ax[1].plot(res.epoch, res['metrics/mAP50-95(B)'], label='mAP50-95')
ax[1].set_title('mAP en validación (' + ' + '.join(CFG.val_splits) + ')'); ax[1].legend()
ax[2].plot(res.epoch, res['metrics/precision(B)'], label='precisión'); ax[2].plot(res.epoch, res['metrics/recall(B)'], label='recall')
ax[2].set_title('Precisión / recall (val)'); ax[2].legend()
plt.tight_layout(); plt.savefig(FIGS / '03_curvas_entrenamiento.png', dpi=130); plt.close('all')

