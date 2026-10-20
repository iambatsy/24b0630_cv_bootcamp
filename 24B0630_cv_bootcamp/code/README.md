# Code and reproduction

Everything was run on a Colab T4 GPU with PyTorch, seed 0. All scripts take
--root (the folder that contains data/ and model.py) and --out (a folder for results).
Training settings are in common.py and are the same for every run: SGD (momentum 0.9,
Nesterov), lr 0.05 with cosine decay, weight decay 5e-4, batch size 128, 60 epochs,
random crop (pad 4) and horizontal flip, ResNet-18 from model.py trained from scratch.
Every script saves a checkpoint each epoch and resumes if it is run again.

Run the commands in this order.

Task 1 and Task 2 (clean reference, noisy baseline, per-epoch dynamics):

    python t1_t2.py --root ROOT --out OUT

This writes t1_clean_ref.csv, t1_noisy_ce.csv, t2_dynamics.csv, the confusion matrix and
a first t2_early_stop.csv. Then pick the early stopping epoch with the final rule
(noisy validation accuracy only) and rewrite t2_early_stop.csv:

    python t2_pick_epoch.py OUT

Task 3 (noise scores, uses the saved per-epoch losses from the Task 1 noisy run):

    python t3.py --root ROOT --out OUT

It compares four scores (loss at the early stop epoch, mean loss over training,
5-fold cross-validated confidence with 30 epochs per fold, and a rank average).
Clean labels are used only to measure AUROC and precision@k. The best by AUROC,
mean loss over training, is written to t3_noise_scores.csv.

Task 4 (three methods, one run each):

    python t4.py --root ROOT --out OUT --method gce
    python t4.py --root ROOT --out OUT --method filter
    python t4.py --root ROOT --out OUT --method coteach

gce is the generalized cross entropy loss (q = 0.7). filter drops the 40 percent of
training images with the highest Task 3 scores and trains on the rest. coteach trains two
networks that pick small loss samples for each other (forget rate ramped to 0.4 over
10 epochs), and its predictions average the two networks. The epoch of each method is
chosen with noisy validation accuracy only, using the same rule as Task 2.

Task 5 (synthetic labels):

    python t4.py --root ROOT --out OUT --method ce --label_type synthetic
    python t4.py --root ROOT --out OUT --method coteach --label_type synthetic

These write t5_baseline_synth.csv and t5_best_synth.csv. Epoch selection uses the
synthetic validation labels only.

The validation numbers in results.json were copied from the summary printed by each
script (and the *_summary.json files in OUT).
