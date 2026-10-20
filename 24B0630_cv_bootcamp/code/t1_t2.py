"""Task 1 (clean reference + noisy CE baseline) and Task 2 (training dynamics, early stopping).

Colab usage:
    python t1_t2.py --root /content/drive/MyDrive/CV --out /content/drive/MyDrive/cv_out
Add --subset 1000 --epochs 3 for a quick smoke test.
Safe to re-run after a disconnect: it resumes from the last finished epoch.
"""
import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=EPOCHS)
ap.add_argument("--subset", type=int, default=None)
a = ap.parse_args()

pred_dir = os.path.join(a.out, "predictions"); os.makedirs(pred_dir, exist_ok=True)
D = load_data(a.root, a.subset)
ynoisy, yclean = D["tr_noisy_label"], D["tr_clean_label"]
mis = ynoisy != yclean            # mislabeled subset (measurement only)
cor = ~mis

# ---------------- Task 1: noise rate + confusion matrix ----------------
noise_rate = float(mis.mean())
cm = pd.crosstab(pd.Series(yclean, name="clean"), pd.Series(ynoisy, name="noisy"))
cm.to_csv(os.path.join(a.out, "confusion_clean_vs_noisy.csv"))
print(f"noise rate (train) = {noise_rate:.4f}")
off = cm.values.copy(); np.fill_diagonal(off, 0)
top = sorted([(off[i, j], i, j) for i in range(10) for j in range(10)], reverse=True)[:8]
print("top clean->noisy confusions (count, clean, noisy):", [(int(c), i, j) for c, i, j in top])

# ---------------- Task 1: clean-label reference ----------------
ref_pred_path = os.path.join(pred_dir, "t1_clean_ref.csv")
if not os.path.exists(ref_pred_path):
    m = train_ce(D["Xtr"], yclean, epochs=a.epochs, tag="clean_ref",
                 ckpt_path=os.path.join(a.out, "ckpt_clean_ref.pt"))
    write_preds(ref_pred_path, predict_logits(m, D["Xte"]).argmax(1).cpu().numpy())
    va_pred = predict_logits(m, D["Xva"]).argmax(1).cpu().numpy()
    ref_val_noisy = float((va_pred == D["va_noisy_label"]).mean())
    json.dump(dict(val_acc_noisy_labels__clean_ref_model=ref_val_noisy,
                   val_acc_clean_labels__clean_ref_model=float((va_pred == D["va_clean_label"]).mean())),
              open(os.path.join(a.out, "t1_clean_ref_metrics.json"), "w"), indent=1)
    print("clean ref: val acc vs noisy val labels =", ref_val_noisy)
    del m

# ---------------- Task 1 + 2: noisy-label baseline with per-epoch logging ----------------
ep_dir = os.path.join(a.out, "noisy_run"); os.makedirs(ep_dir, exist_ok=True)
Xtr, ytr_t = D["Xtr"], torch.as_tensor(ynoisy, device=DEV).long()
vn, vc = D["va_noisy_label"], D["va_clean_label"]

def on_epoch(ep, model):
    lg = predict_logits(model, Xtr)                      # eval mode, no augmentation
    loss = F.cross_entropy(lg, ytr_t, reduction="none").cpu().numpy()
    tp = lg.argmax(1).cpu().numpy()
    vp = predict_logits(model, D["Xva"]).argmax(1).cpu().numpy()
    tep = predict_logits(model, D["Xte"]).argmax(1).cpu().numpy()
    np.savez(os.path.join(ep_dir, f"ep_{ep:02d}.npz"),
             train_loss=loss.astype(np.float16), train_pred=tp.astype(np.int8),
             val_pred=vp.astype(np.int8), test_pred=tep.astype(np.int8))

if len([f for f in os.listdir(ep_dir) if f.endswith(".npz")]) < a.epochs:
    m = train_ce(Xtr, ynoisy, epochs=a.epochs, on_epoch=on_epoch, tag="noisy_ce",
                 ckpt_path=os.path.join(a.out, "ckpt_noisy_ce.pt"))

# assemble the dynamics from the saved per-epoch files
rows = []
for ep in range(1, a.epochs + 1):
    z = np.load(os.path.join(ep_dir, f"ep_{ep:02d}.npz"))
    tp, vp = z["train_pred"].astype(int), z["val_pred"].astype(int)
    rows.append(dict(
        epoch=ep,
        train_acc_correct=float((tp[cor] == ynoisy[cor]).mean()),
        train_acc_mislabeled=float((tp[mis] == ynoisy[mis]).mean()),   # vs the WRONG (noisy) label = memorization
        val_acc_noisy=float((vp == vn).mean()),
        val_acc_clean=float((vp == vc).mean()),
        train_acc_mislabeled_vs_clean=float((tp[mis] == yclean[mis]).mean()),  # extra, for the report only
    ))
dyn = pd.DataFrame(rows)
dyn.to_csv(os.path.join(a.out, "dynamics_full.csv"), index=False)
dyn.iloc[:, :5].to_csv(os.path.join(pred_dir, "t2_dynamics.csv"), index=False)

# final-epoch noisy CE test predictions (Task 1)
last = np.load(os.path.join(ep_dir, f"ep_{a.epochs:02d}.npz"))
write_preds(os.path.join(pred_dir, "t1_noisy_ce.csv"), last["test_pred"])

# ---------------- Task 2: early stopping on NOISY val accuracy only ----------------
v = dyn["val_acc_noisy"].values
smooth = np.convolve(v, np.ones(3) / 3, mode="same"); smooth[0], smooth[-1] = v[0], v[-1]
es = int(np.argmax(smooth)) + 1
es_raw = int(np.argmax(v)) + 1
z = np.load(os.path.join(ep_dir, f"ep_{es:02d}.npz"))
write_preds(os.path.join(pred_dir, "t2_early_stop.csv"), z["test_pred"])

# measurement-only markers for memorization (clean labels used only to measure)
peak_gen = int(dyn["train_acc_mislabeled_vs_clean"].values.argmax()) + 1
peak_clean_val = int(dyn["val_acc_clean"].values.argmax()) + 1
summary = dict(noise_rate_train=noise_rate,
               val_acc_noisy__noisy_ce_final=float(v[-1]),
               early_stop_epoch=es, early_stop_epoch_unsmoothed=es_raw,
               val_acc_noisy_at_early_stop=float(v[es - 1]),
               epoch_mislabeled_acc_vs_clean_peaks=peak_gen,
               epoch_clean_val_acc_peaks=peak_clean_val)
json.dump(summary, open(os.path.join(a.out, "t1_t2_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

# plain black/white plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(dyn.epoch, dyn.train_acc_correct, "k-", label="train, correctly labeled")
plt.plot(dyn.epoch, dyn.train_acc_mislabeled, "k--", label="train, mislabeled (vs noisy label)")
plt.plot(dyn.epoch, dyn.val_acc_noisy, "k:", label="val (noisy labels)")
plt.plot(dyn.epoch, dyn.val_acc_clean, "k-.", label="val (clean labels)")
plt.axvline(es, color="gray", lw=0.8)
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(os.path.join(a.out, "dynamics.png"), dpi=150)
