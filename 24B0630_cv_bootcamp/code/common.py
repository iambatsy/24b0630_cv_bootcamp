"""Shared helpers for all tasks. Same optimizer/schedule/augmentation everywhere."""
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ResNet18

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# training settings, kept identical across every run
BATCH = 128
LR = 0.05
WD = 5e-4
MOMENTUM = 0.9
EPOCHS = 60
SEED = 0


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_data(root, subset=None):
    """root = folder that contains data/. Returns dict of numpy arrays + torch image tensors."""
    d = os.path.join(root, "data")
    import pandas as pd
    lab = pd.read_csv(os.path.join(d, "labels.csv")).sort_values("index")
    tr = np.loadtxt(os.path.join(d, "train_idx.txt"), dtype=int)
    va = np.loadtxt(os.path.join(d, "val_idx.txt"), dtype=int)
    if subset:  # smoke test only
        tr, va = tr[:subset], va[:max(subset // 5, 100)]
    X = np.load(os.path.join(d, "train_images.npy"))
    Xte = np.load(os.path.join(d, "test_images.npy"))
    if subset:
        Xte = Xte[:subset]
    out = dict(train_idx=tr, val_idx=va)
    for k in ["noisy_label", "synthetic_label", "clean_label"]:
        arr = lab[k].values
        out["tr_" + k] = arr[tr]
        out["va_" + k] = arr[va]
    # normalisation stats come from the provided training images only
    xt = X[tr].astype(np.float32) / 255.0
    mean = xt.mean((0, 1, 2)); std = xt.std((0, 1, 2))
    def prep(a):
        a = (a.astype(np.float32) / 255.0 - mean) / std
        return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(DEV)
    out["Xtr"] = prep(X[tr]); out["Xva"] = prep(X[va]); out["Xte"] = prep(Xte)
    return out


def augment(x):
    """random crop (pad 4) + horizontal flip, on the GPU."""
    B = x.size(0)
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=x.device)
    j = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar)[:, None, :, None]
    cols = (j[:, None] + ar)[:, None, None, :]
    b = torch.arange(B, device=x.device)[:, None, None, None]
    c = torch.arange(3, device=x.device)[None, :, None, None]
    out = xp[b, c, rows, cols]
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


@torch.no_grad()
def predict_logits(model, X, bs=1000):
    model.eval()
    outs = []
    for k in range(0, X.size(0), bs):
        with torch.autocast(DEV, enabled=(DEV == "cuda")):
            outs.append(model(X[k:k + bs]).float())
    return torch.cat(outs)


def train_ce(X, y, epochs=EPOCHS, seed=SEED, on_epoch=None, loss_fn=None,
             ckpt_path=None, tag=""):
    """Plain training loop. loss_fn(logits, y) defaults to cross-entropy.
    on_epoch(epoch, model) is called after each epoch (epoch is 1-based).
    If ckpt_path is given, training resumes from it after a Colab disconnect."""
    seed_all(seed)
    model = ResNet18(10).to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WD, nesterov=True)
    n = X.size(0)
    steps_per_ep = n // BATCH  # drop last partial batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep)
    scaler = torch.amp.GradScaler(enabled=(DEV == "cuda"))
    y = torch.as_tensor(y, device=DEV).long()
    loss_fn = loss_fn or (lambda lg, t: F.cross_entropy(lg, t))
    start = 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start = ck["epoch"]
        print(f"[{tag}] resumed from epoch {start}")
    for ep in range(start + 1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for s in range(steps_per_ep):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            xb = augment(X[idx]).contiguous(memory_format=torch.channels_last)
            with torch.autocast(DEV, enabled=(DEV == "cuda")):
                logits = model(xb)
            loss = loss_fn(logits.float(), y[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() if s % 50 == 0 else 0
        if on_epoch:
            on_epoch(ep, model)
        if ckpt_path:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), scaler=scaler.state_dict(),
                            epoch=ep), ckpt_path)
        print(f"[{tag}] epoch {ep}/{epochs} done", flush=True)
    return model


def write_preds(path, preds):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("id,pred\n")
        for i, p in enumerate(preds):
            f.write(f"{i},{int(p)}\n")
