import csv

data = []
with open(r'C:\Users\ahmet\Downloads\results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epoch_val = row.get('epoch', '').strip()
        if epoch_val:
            data.append(row)

print("=== EGITIM OZETI ===")
print(f"Toplam epoch: {len(data)}")

print("\n=== MAP50-95 ILERLEME ===")
for d in data:
    e = int(d['epoch'].strip())
    m50 = float(d['metrics/mAP50(B)'].strip())
    m5095 = float(d['metrics/mAP50-95(B)'].strip())
    tbl = float(d['train/box_loss'].strip())
    vbl = float(d['val/box_loss'].strip())
    tcl = float(d['train/cls_loss'].strip())
    vcl = float(d['val/cls_loss'].strip())
    prec = float(d['metrics/precision(B)'].strip())
    rec = float(d['metrics/recall(B)'].strip())
    print(f"E{e:2d}: mAP50={m50:.4f} mAP50-95={m5095:.4f} P={prec:.3f} R={rec:.3f} | tBox={tbl:.4f} vBox={vbl:.4f} | tCls={tcl:.4f} vCls={vcl:.4f}")

metrics = [(int(d['epoch'].strip()), float(d['metrics/mAP50(B)'].strip()), float(d['metrics/mAP50-95(B)'].strip())) for d in data]
best_ep, _, best_m5095 = max(metrics, key=lambda x: x[2])
print(f"\nEn iyi mAP50-95: {best_m5095:.4f} (epoch {best_ep})")
best_ep50, best_m50, _ = max(metrics, key=lambda x: x[1])
print(f"En iyi mAP50:    {best_m50:.4f} (epoch {best_ep50})")

# Improvement rates
print("\n=== PLATO ANALIZI ===")
prev_m = None
for e, m50, m5095 in metrics:
    if prev_m is not None:
        delta = m5095 - prev_m
        marker = " <<<< PLATO" if abs(delta) < 0.001 else ""
        if e >= 40:
            print(f"E{e}: mAP50-95={m5095:.4f}  delta={delta:+.4f}{marker}")
    prev_m = m5095

# Overfitting check
print("\n=== OVERFITTING KONTROLU ===")
print("(val_cls_loss train_cls_loss'tan ne kadar buyuk?)")
for d in data:
    e = int(d['epoch'].strip())
    tcl = float(d['train/cls_loss'].strip())
    vcl = float(d['val/cls_loss'].strip())
    tbl = float(d['train/box_loss'].strip())
    vbl = float(d['val/box_loss'].strip())
    cls_gap = vcl - tcl
    box_gap = vbl - tbl
    if e in [6, 15, 25, 35, 45, 55, 57]:
        print(f"E{e:2d}: cls_gap={cls_gap:.4f}  box_gap={box_gap:.4f}  (vCls={vcl:.4f})")

# val_cls_loss trend
print("\n=== VAL_CLS_LOSS TRENDI (overfitting gostergesi) ===")
for d in data:
    e = int(d['epoch'].strip())
    vcl = float(d['val/cls_loss'].strip())
    if e >= 45:
        print(f"E{e}: val_cls_loss={vcl:.5f}")
