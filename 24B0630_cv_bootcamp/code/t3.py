"""Task 3: score every training image, higher = more likely that its noisy label is wrong.

Needs the Task 1 noisy run (cv_out/noisy_run/ep_XX.npz) already finished.
Colab usage:
    python t3.py --root /content/drive/MyDrive/CV --out /content/drive/MyDrive/cv_out
Add --subset 1000 --epochs 3 for a smoke test (use a different --out for that).
Resumes fold by fold if Colab disconnects.
"""
import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--t1_epochs", type=int, default=60)   # epochs of the Task 1 noisy run
ap.add_argument("--early_stop", type=int, default=20)  # Task 2 early-stop epoch
ap.add_argument("--epochs", type=int, default=30)      # epochs per CV fold network
ap.add_argument("--folds", type=int, default=5)        # Task 3 budget: at most 5 runs
ap.add_argument("--subset", type=int, default=None)
a = ap.parse_args()
assert a.folds <= 5 and a.epochs <= 60

D = load_data(a.root, a.subset)
ynoisy, yclean = D["tr_noisy_label"], D["tr_clean_label"]
n = len(ynoisy)
mis = (ynoisy != yclean)                  # used ONLY to measure the scores
k = int(mis.sum())
cand_dir = os.path.join(a.out, "t3_candidates"); os.makedirs(cand_dir, exist_ok=True)
pred_dir = os.path.join(a.out, "predictions"); os.makedirs(pred_dir, exist_ok=True)

# ---- signals A and B: free, from the Task 1 noisy run (no new training) ----
ep_dir = os.path.join(a.out, "noisy_run")
traj = np.stack([np.load(os.path.join(ep_dir, f"ep_{e:02d}.npz"))["train_loss"].astype(np.float32)
                 for e in range(1, a.t1_epochs + 1)])           # (epochs, n)
sig = {}
sig["A_loss_at_early_stop"] = traj[a.early_stop - 1]
sig["B_mean_loss_over_training"] = traj.mean(0)

# ---- signal C: 5-fold cross-validated confidence in the noisy label ----
rng = np.random.RandomState(SEED)
fold_of = rng.permutation(n) % a.folds
oof = np.zeros((n, 10), dtype=np.float32)
for f in range(a.folds):
    fp = os.path.join(cand_dir, f"oof_fold{f}.npy")
    te = np.where(fold_of == f)[0]; trn = np.where(fold_of != f)[0]
    if os.path.exists(fp):
        oof[te] = np.load(fp); continue
    m = train_ce(D["Xtr"][torch.as_tensor(trn, device=DEV)], ynoisy[trn], epochs=a.epochs,
                 seed=SEED + 1 + f, tag=f"cv_fold{f}",
                 ckpt_path=os.path.join(a.out, f"ckpt_t3_fold{f}.pt"))
    p = F.softmax(predict_logits(m, D["Xtr"][torch.as_tensor(te, device=DEV)]), 1).cpu().numpy()
    np.save(fp, p); oof[te] = p
    del m
p_label = oof[np.arange(n), ynoisy]
sig["C_cv_neg_log_conf"] = -np.log(np.clip(p_label, 1e-8, 1.0))

# ---- signal D: rank-average of A, B, C ----
r = lambda s: rankdata(s) / len(s)
sig["D_rank_average"] = (r(sig["A_loss_at_early_stop"]) + r(sig["B_mean_loss_over_training"])
                         + r(sig["C_cv_neg_log_conf"])) / 3.0

# ---- measure with clean labels (the Task 3 'own check' the rules allow) ----
res = {}
for name, s in sig.items():
    topk = np.argsort(-s)[:k]
    res[name] = dict(auroc=float(roc_auc_score(mis, s)), precision_at_k=float(mis[topk].mean()))
    pd.DataFrame({"index": D["train_idx"], "score": s}).to_csv(os.path.join(cand_dir, f"{name}.csv"), index=False)
res["_k_true_mislabeled"] = k
best = max([x for x in sig], key=lambda x: res[x]["auroc"])
res["_best_by_auroc"] = best
json.dump(res, open(os.path.join(a.out, "t3_summary.json"), "w"), indent=1)
print(f"k (true mislabeled in train) = {k}; random baseline AUROC 0.5, precision@k {k / n:.3f}")
for name in sig:
    print(f"{name:30s} AUROC {res[name]['auroc']:.4f}  precision@k {res[name]['precision_at_k']:.4f}")
print("best by AUROC:", best)

# final file for submission
pd.DataFrame({"index": D["train_idx"], "score": sig[best]}).to_csv(
    os.path.join(pred_dir, "t3_noise_scores.csv"), index=False)

# ---- failure-mode info for Q3 (best signal) ----
s = sig[best]
rank_pct = rankdata(s) / n                       # low = scored as 'looks clean'
missed = mis & (rank_pct < 0.5)                  # mislabeled but in the lower half of scores
pairs = pd.crosstab(pd.Series(yclean[missed], name="clean"), pd.Series(ynoisy[missed], name="noisy")).stack()
print("\nmislabeled images the best signal misses (lower half), top (clean, noisy) pairs:")
print(pairs.sort_values(ascending=False).head(6).to_string())
fp_mask = (~mis) & (rank_pct > 0.8)
print("clean-labeled images scored in top 20% (false alarms) by true class:",
      np.bincount(yclean[fp_mask], minlength=10).tolist())
print("fraction of each noisy-label class in the top-k flagged set:",
      np.round(np.bincount(ynoisy[np.argsort(-s)[:k]], minlength=10) / k, 3).tolist())
