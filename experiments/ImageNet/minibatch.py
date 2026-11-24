#!/usr/bin/env python3
"""
Mini-batch OT (MB) alignment of ImageNet ResNet50 embeddings, evaluated exactly
like run_omt.py / hiref.py (shared split, classifier, metrics).

Method (mini-batch OT without replacement, as in src/validation.py):
  * shuffle X and Y independently, cut both into aligned batches of size B
  * per batch, solve OT on the squared-Euclidean cost C_b:
      - "sinkhorn": entropic OT (ot.sinkhorn_log, reg on C_b/C_b.max()), map is the
        within-batch barycentric projection T(x_i) = sum_j P_ij y_j / sum_j P_ij
      - "emd":      exact OT (ot.emd) -> permutation, T(x_i) = y_sigma(i)
  * the global plan is block-diagonal (1/n_batches weight per block).

Usage:
    python minibatch.py --per-class 366 --batch-size 1024 --method sinkhorn
Output:
    <results_dir>/imagenet_mb_<method>_B<B>_n<n>_c<n_classes>.json
"""
import argparse, json, os, sys, time
import numpy as np
import ot
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import PATHS, add_code_dirs
add_code_dirs()
from run_omt import (EMB_PATH, OUT_DIR, RESNET_PATH, load_classifier,
                          load_source_target, nearest_target, predict, topk_hit)


def nearest_target_gpu(X_hat, Y_cpu, device, chunk=256):
    """Nearest target for each T(x); Y lives on the GPU, X_hat streamed in chunks."""
    Y_g = Y_cpu.to(device)
    y_sq = (Y_g * Y_g).sum(1)
    idx, dist = [], []
    for s in range(0, X_hat.shape[0], chunk):
        xb = X_hat[s:s + chunk].to(device)
        d = (xb * xb).sum(1, keepdim=True) + y_sq[None, :] - 2.0 * (xb @ Y_g.T)
        b = d.min(1)
        idx.append(b.indices.cpu()); dist.append(b.values.clamp(min=0).cpu())
    del Y_g
    return torch.cat(idx).numpy(), torch.cat(dist).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=10000)
    p.add_argument("--per-class", type=int, default=None)
    p.add_argument("--n-classes", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--method", default="sinkhorn", choices=["sinkhorn", "emd"])
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--emb", default=EMB_PATH)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"On device: {device}", flush=True)
    torch.manual_seed(args.seed)

    X, class_X, Y, class_Y = load_source_target(args.emb, args.n_pairs, args.seed,
                                                per_class=args.per_class,
                                                n_classes_subset=args.n_classes)
    n, d = X.shape
    n_classes = len(np.unique(class_X))
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)   # CPU; batches go to GPU
    Y_t = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
    clf = load_classifier(RESNET_PATH, d, device)

    rng = np.random.default_rng(args.seed + 7)
    perm_x, perm_y = rng.permutation(n), rng.permutation(n)
    B = min(args.batch_size, n)
    X_hat = torch.empty_like(X_t)
    match = np.empty(n, dtype=np.int64)        # argmax target per source
    plan_cost, plan_class_acc = 0.0, 0.0
    t0 = time.time()
    starts = list(range(0, n, B))
    for k, s in enumerate(starts):
        ix = torch.as_tensor(perm_x[s:s + B])
        iy = torch.as_tensor(perm_y[s:s + B])
        xb, yb = X_t[ix].to(device).double(), Y_t[iy].to(device).double()
        C = torch.cdist(xb, yb) ** 2
        nb = C.shape[0]
        a = torch.full((nb,), 1.0 / nb, dtype=C.dtype, device=device)
        if args.method == "sinkhorn":
            P = ot.bregman.sinkhorn_log(a, a, C / C.max(), args.reg,
                                        numItermax=100000, stopThr=1e-6)
        else:
            P = ot.emd(a, a, C)
        P = P / P.sum()
        row = P.sum(1, keepdim=True)
        X_hat[ix] = ((P / row) @ yb).float().cpu()
        match[ix.cpu().numpy()] = iy.cpu().numpy()[P.argmax(1).cpu().numpy()]
        w = nb / n
        plan_cost += w * float((P * C).sum())
        same = torch.as_tensor(class_X[ix.cpu().numpy()][:, None] ==
                               class_Y[iy.cpu().numpy()][None, :], device=device)
        plan_class_acc += w * float((P * same).sum())
        if k % 50 == 0:
            print(f"  batch {k+1}/{len(starts)}  {time.time()-t0:.1f}s", flush=True)
    solve_time = time.time() - t0
    print(f"solved in {solve_time:.1f}s", flush=True)

    displacement = float(torch.norm(X_t - X_hat, dim=1).mean())
    nn_idx, nn_sq = nearest_target_gpu(X_hat, Y_t, device)
    pred_X, top5_X = predict(clf, X, device)
    pred_Y, _ = predict(clf, Y, device)
    pred_hat, top5_hat = predict(clf, X_hat.cpu().numpy(), device)

    m = {
        "method": f"minibatch_{args.method}", "batch_size": B, "reg": args.reg,
        "n_per_side": int(n), "n_classes": int(n_classes),
        "per_class_per_side": int(n // n_classes), "dim": int(d), "seed": args.seed,
        "transport_cost_plan": plan_cost,
        "mean_displacement_L2": displacement,
        "mean_dist_to_nearest_target": float(np.sqrt(nn_sq).mean()),
        "matched_class_accuracy": float(np.mean(class_Y[match] == class_X)),
        "nearest_neighbour_class_accuracy": float(np.mean(class_Y[nn_idx] == class_X)),
        "plan_weighted_class_accuracy": plan_class_acc,
        "clf_top1_source": float(np.mean(pred_X == class_X)),
        "clf_top1_target": float(np.mean(pred_Y == class_Y)),
        "clf_top1_transported": float(np.mean(pred_hat == class_X)),
        "clf_top5_source": topk_hit(top5_X, class_X),
        "clf_top5_transported": topk_hit(top5_hat, class_X),
        "label_preservation_top1": float(np.mean(pred_hat == pred_X)),
        "label_preservation_top5": topk_hit(top5_hat, pred_X),
        "n_distinct_targets_matched": int(len(np.unique(match))),
        "solve_time_s": solve_time, "n_batches": len(starts),
    }
    print(json.dumps(m, indent=2))
    out = args.out or os.path.join(
        OUT_DIR, f"imagenet_mb_{args.method}_B{B}_n{n}_c{n_classes}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Metrics written to {out}", flush=True)


if __name__ == "__main__":
    main()
