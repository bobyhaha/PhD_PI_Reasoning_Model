import argparse, glob, json, os
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--runs_dir', default='runs'); ap.add_argument('--out', default='comparison.png'); args = ap.parse_args()
paths = glob.glob(os.path.join(args.runs_dir, '**', 'metrics.jsonl'), recursive=True)
if not paths:
    print('No metrics.jsonl found'); raise SystemExit
for path in paths:
    label = os.path.relpath(os.path.dirname(path), args.runs_dir); xs=[]; acc=[]
    for line in open(path):
        m=json.loads(line); xs.append(m['epoch']); acc.append(m.get('val_acc', m.get('val_exact_acc', m.get('val_token_acc', 0.0))))
    plt.plot(xs, acc, label=label)
plt.xlabel('epoch'); plt.ylabel('validation accuracy'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(args.out, dpi=200); print(f'Saved {args.out}')
