"""Generador de un dataset sintético con el mismo formato que CCT-20 (para `--smoke`).
Rectángulos de color sobre fondos distintos por cámara; 4 clases; 5 particiones (train, cis_val, cis_test, trans_val, trans_test).
Sirve SOLO para verificar que el pipeline corre de extremo a extremo; sus métricas no significan nada."""
import numpy as np, cv2, json, random, datetime
from pathlib import Path


def make_synthetic(root):
    root = Path(root); (root/'images').mkdir(parents=True, exist_ok=True); (root/'annotations').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0); random.seed(0)
    names = ['bobcat','coyote','deer','car']; cat_ids = {'empty':30,'bobcat':6,'coyote':9,'deer':11,'car':33}
    colors = {'bobcat':(40,40,220),'coyote':(40,200,40),'deer':(220,60,60),'car':(200,200,40)}
    W,H = 320,240
    plan = {'train': (['L1','L2','L3'], 60), 'cis_val': (['L1','L2','L3'], 4), 'cis_test': (['L1','L2','L3'], 8),
            'trans_val': (['L8'], 10), 'trans_test': (['L9','L10'], 18)}
    locbg = {l: rng.integers(60,200) for l in ['L1','L2','L3','L8','L9','L10']}
    iid = 0
    import datetime
    for split,(locs,n) in plan.items():
        imgs, anns, aid = [], [], 0
        for loc in locs:
            t0 = datetime.datetime(2012,5,1,6,0,0) + datetime.timedelta(days=random.randint(0,40))
            for k in range(n):
                iid += 1; seq = f'{split}-{loc}-{k//3}'
                bg = np.clip(rng.normal(locbg[loc], 25, (H,W,3)),0,255).astype(np.uint8)
                fn = f'img_{iid:05d}.jpg'
                nanim = random.choices([0,1,2],[0.25,0.6,0.15])[0]
                boxes=[]
                for _ in range(nanim):
                    c = random.choice(names); w = random.randint(30,110); h = random.randint(25,80)
                    x = random.randint(0,W-w); y = random.randint(0,H-h)
                    cv2.rectangle(bg,(x,y),(x+w,y+h),colors[c],-1); boxes.append((c,x,y,w,h))
                cv2.imwrite(str(root/'images'/fn), bg)
                t = t0 + datetime.timedelta(minutes=17*k + random.randint(0,3)*60*(k//3))
                imgs.append(dict(id=f'{split}{iid}', file_name=fn, width=W, height=H, location=loc, seq_id=seq, frame_num=k%3, seq_num_frames=3, date_captured=t.strftime('%Y-%m-%d %H:%M:%S')))
                if not boxes:
                    aid+=1; anns.append(dict(id=aid, image_id=f'{split}{iid}', category_id=30))
                for (c,x,y,w,h) in boxes:
                    aid+=1; a=dict(id=aid, image_id=f'{split}{iid}', category_id=cat_ids[c], bbox=[x,y,w,h])
                    if split=='train' and random.random()<0.08: a.pop('bbox')
                    anns.append(a)
        cats = [dict(id=v,name=k) for k,v in cat_ids.items()]
        json.dump(dict(images=imgs, annotations=anns, categories=cats), open(root/'annotations'/f'{split}_annotations.json','w'))
    return iid

