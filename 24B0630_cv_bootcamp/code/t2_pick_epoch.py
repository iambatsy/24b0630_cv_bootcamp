"""Task 2: choose the early stopping epoch using NOISY val accuracy only,
then rewrite predictions/t2_early_stop.csv from the saved per-epoch test predictions.
Rule: smooth noisy val accuracy with a 5 epoch centred mean, then take the earliest
epoch whose smoothed value is within 0.007 of the best smoothed value.
Usage: python t2_pick_epoch.py <out folder>"""
import sys
import numpy as np
import pandas as pd

out = sys.argv[1]
d = pd.read_csv(f"{out}/dynamics_full.csv")
v = d["val_acc_noisy"]
sm = v.rolling(5, center=True, min_periods=1).mean()
es = int(d["epoch"][(sm >= sm.max() - 0.007).values].iloc[0])
print("early stop epoch:", es, "| noisy val acc there:", round(float(v[es - 1]), 4))
z = np.load(f"{out}/noisy_run/ep_{es:02d}.npz")
with open(f"{out}/predictions/t2_early_stop.csv", "w") as f:
    f.write("id,pred\n")
    for i, p in enumerate(z["test_pred"]):
        f.write(f"{i},{int(p)}\n")
