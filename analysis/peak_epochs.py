import os, re, glob

base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines/results/nnUNet_results")
for d in sorted(glob.glob(f"{base}/Dataset*/nnUNetTrainer150__nnUNetPlans__2d/fold_0")):
    dataset = re.search(r'Dataset[^/]*', d).group()
    log = glob.glob(f"{d}/training_log_*.txt")[0]
    current_epoch = None
    best_epoch = None
    best_dice = 0
    for line in open(log):
        m = re.search(r'Epoch (\d+)$', line.strip())
        if m:
            current_epoch = int(m.group(1))
        m = re.search(r'New best EMA pseudo Dice: ([\d.]+)', line)
        if m:
            best_epoch = current_epoch
            best_dice = float(m.group(1))
    print(f"{dataset}: Peak at epoch {best_epoch}/150, EMA Dice {best_dice:.4f}")
