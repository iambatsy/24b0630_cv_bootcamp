"""Task 4 (and later Task 5): robust training methods, one method per run.

    --method gce       generalized cross entropy loss (q=0.7)
    --method filter    drop the 40% highest Task 3 scores, then train normally
    --method coteach   co-teaching, 2 networks, small-loss exchange
    --method ce        plain cross entropy (only needed as the Task 5 baseline)

Colab usage (run once per method):
    python t4.py --root /content/drive/MyDrive/CV --out /content/drive/MyDrive/cv_out --method gce
Every method uses the same optimizer, schedule, augmentation and 60 epochs as the Task 1 baseline.
The epoch is chosen with NOISY val accuracy only (same rule as Task 2). Resumes after a disconnect.
"""
import argparse, os, json, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import *
from model import ResNet18

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--method", required=True, choices=["gce", "filter", "coteach", "ce"])
ap.add_argument("--label_type", default="noisy", choices=["noisy", "synthetic"])
ap.add_argument("--epochs", type=int, default=EPOCHS)
ap.add_argument("--q", type=float, default=0.7)          # GCE
ap.add_argument("--drop", type=float, default=0.40)      # filter: fraction removed
ap.add_argument("--tau", type=float, default=0.40)       # co-teaching: final forget rate
ap.add_argument("--Tk", type=int, default=10)            # co-teaching: epochs to ramp the forget rate
ap.add_argument("--scores", default=None)                # filter: Task 3 scores csv
ap.add_argument("--subset", type=int, default=None)
a = ap.parse_args()

D = load_data(a.root, a.subset)
lt = a.label_type
ytr, yva = D[f"tr_{lt}_label"], D[f"va_{lt}_label"]
yva_clean = D["va_clean_label"]                       # logging/measurement only
tag = f"t4_{a.method}" + ("" if lt == "noisy" else "_synth")
run_dir = os.path.join(a.out, tag); os.makedirs(run_dir, exist_ok=True)
Xtr = D["Xtr"]

def save_epoch(ep, val_pred, test_pred, extra=None):
    np.savez(os.path.join(run_dir, f"ep_{ep:02d}.npz"), val_pred=val_pred.astype(np.int8),
             test_pred=test_pred.astype(np.int8), **(extra or {}))

def done_epochs():
    return len(glob.glob(os.path.join(run_dir, "ep_*.npz")))

def gce_loss(logits, y, q=a.q):
    p = F.softmax(logits, 1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
    return ((1.0 - p ** q) / q).mean()

# ------------------------------------------------------------------ training
if done_epochs() < a.epochs:
    ck = os.path.join(a.out, f"ckpt_{tag}.pt")
    if a.method in ("gce", "filter", "ce"):
        X, y = Xtr, ytr
        if a.method == "filter":
            if lt != "noisy" and a.scores is None:
                raise SystemExit("filter needs Task 3 scores computed for these labels; pass --scores")
            sp = a.scores or os.path.join(a.out, "predictions", "t3_noise_scores.csv")
            sc = pd.read_csv(sp).set_index("index").loc[D["train_idx"], "score"].values
            keep = sc <= np.quantile(sc, 1.0 - a.drop)
            print(f"filter: keeping {keep.sum()} of {len(keep)} images")
            kt = torch.as_tensor(np.where(keep)[0], device=DEV)
            X, y = Xtr[kt], ytr[keep]
        def on_epoch(ep, model):
            vp = predict_logits(model, D["Xva"]).argmax(1).cpu().numpy()
            tp = predict_logits(model, D["Xte"]).argmax(1).cpu().numpy()
            save_epoch(ep, vp, tp)
        train_ce(X, y, epochs=a.epochs, on_epoch=on_epoch, tag=tag, ckpt_path=ck,
                 loss_fn=(gce_loss if a.method == "gce" else None))
    else:
        # ---------------- co-teaching with two networks ----------------
        seed_all(SEED)
        nets = [ResNet18(10).to(DEV).to(memory_format=torch.channels_last) for _ in range(2)]
        opts = [torch.optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD, nesterov=True)
                for m in nets]
        n = Xtr.size(0); spe = n // BATCH
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=a.epochs * spe) for o in opts]
        scalers = [torch.amp.GradScaler(enabled=(DEV == "cuda")) for _ in range(2)]
        yt = torch.as_tensor(ytr, device=DEV).long()
        start = 0
        if os.path.exists(ck):
            c = torch.load(ck, map_location=DEV)
            for i in range(2):
                nets[i].load_state_dict(c["nets"][i]); opts[i].load_state_dict(c["opts"][i])
                scheds[i].load_state_dict(c["scheds"][i]); scalers[i].load_state_dict(c["scalers"][i])
            start = c["epoch"]; print(f"[{tag}] resumed from epoch {start}")
        for ep in range(start + 1, a.epochs + 1):
            rate = a.tau * min((ep - 1) / a.Tk, 1.0)
            for m in nets: m.train()
            perm = torch.randperm(n, device=DEV)
            for s in range(spe):
                idx = perm[s * BATCH:(s + 1) * BATCH]
                xb = augment(Xtr[idx]).contiguous(memory_format=torch.channels_last); yb = yt[idx]
                with torch.autocast(DEV, enabled=(DEV == "cuda")):
                    lgs = [m(xb) for m in nets]
                per = [F.cross_entropy(l.float(), yb, reduction="none") for l in lgs]
                nk = max(1, int((1.0 - rate) * len(idx)))
                sel = [torch.argsort(p.detach())[:nk] for p in per]     # small-loss samples of each net
                losses = [per[0][sel[1]].mean(), per[1][sel[0]].mean()]  # each net learns from the other's pick
                for i in range(2):
                    opts[i].zero_grad(set_to_none=True)
                    scalers[i].scale(losses[i]).backward()
                    scalers[i].step(opts[i]); scalers[i].update(); scheds[i].step()
            pv = [F.softmax(predict_logits(m, D["Xva"]), 1) for m in nets]
            pt = [F.softmax(predict_logits(m, D["Xte"]), 1) for m in nets]
            vp = (pv[0] + pv[1]).argmax(1).cpu().numpy()                 # average of the two networks
            tp = (pt[0] + pt[1]).argmax(1).cpu().numpy()
            save_epoch(ep, vp, tp, dict(val_acc_net0=float((pv[0].argmax(1).cpu().numpy() == yva).mean()),
                                        val_acc_net1=float((pv[1].argmax(1).cpu().numpy() == yva).mean())))
            torch.save(dict(nets=[m.state_dict() for m in nets], opts=[o.state_dict() for o in opts],
                            scheds=[s_.state_dict() for s_ in scheds], scalers=[s_.state_dict() for s_ in scalers],
                            epoch=ep), ck)
            print(f"[{tag}] epoch {ep}/{a.epochs} done (forget rate {rate:.2f})", flush=True)

# ------------------------------------------------------------------ epoch selection (labels of this run only)
rows = []
for ep in range(1, a.epochs + 1):
    z = np.load(os.path.join(run_dir, f"ep_{ep:02d}.npz")); vp = z["val_pred"].astype(int)
    rows.append(dict(epoch=ep, val_acc=float((vp == yva).mean()), val_acc_clean=float((vp == yva_clean).mean())))
d = pd.DataFrame(rows); d.to_csv(os.path.join(run_dir, "val_curve.csv"), index=False)
v = d["val_acc"]
sm = v.rolling(5, center=True, min_periods=1).mean()
sel_ep = int(d["epoch"][(sm >= sm.max() - 0.007).values].iloc[0])      # same rule as Task 2
z = np.load(os.path.join(run_dir, f"ep_{sel_ep:02d}.npz"))
summ = dict(method=a.method, label_type=lt, selected_epoch=sel_ep,
            val_acc_at_selected=float(v[sel_ep - 1]), val_acc_max=float(v.max()),
            val_acc_final_epoch=float(v.iloc[-1]), val_acc_clean_at_selected=float(d.val_acc_clean[sel_ep - 1]))
json.dump(summ, open(os.path.join(a.out, f"{tag}_summary.json"), "w"), indent=1)
if lt == "noisy" and a.method != "ce":
    write_preds(os.path.join(a.out, "predictions", f"t4_{a.method}.csv"), z["test_pred"])
elif lt == "synthetic":   # Task 5 files: baseline = plain CE, best = the best Task 4 method
    fn = "t5_baseline_synth.csv" if a.method == "ce" else "t5_best_synth.csv"
    write_preds(os.path.join(a.out, "predictions", fn), z["test_pred"])
print(json.dumps(summ, indent=1))

print("\nall methods so far (val acc against the labels each was trained on):")
for f in sorted(glob.glob(os.path.join(a.out, "t4_*_summary.json"))):
    s = json.load(open(f))
    print(f"  {os.path.basename(f)[:-13]:18s} epoch {s['selected_epoch']:2d}  val {s['val_acc_at_selected']:.4f}  (final epoch {s['val_acc_final_epoch']:.4f})")