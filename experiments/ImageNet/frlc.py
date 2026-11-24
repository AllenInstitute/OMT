#!/usr/bin/env python3
"""
FRLC (low-rank OT, src/FRLC.py:FRLC_LR_opt) alignment of ImageNet ResNet50
embeddings, evaluated exactly like run_omt.py / hiref.py.

The squared-Euclidean cost is never materialised: it is passed in factored form
(HR_OT.compute_lr_sqeuclidean_matrix, rank d+2, rescaled as in HiRef).  FRLC
returns P = Q diag(1/g) R^T of rank r.  Solver params default to those HiRef
uses for its own FRLC sub-problems (gamma=90, 15-30 iters, tau_in=1e5).

Transport map: barycentric projection T(x_i) = n * (Q diag(1/g) (R^T Y))_i.
`matched_class_accuracy` uses the row-argmax of P (computed in chunks).

Usage:
    python frlc.py --per-class 366 --rank 1000
Output:
    <results_dir>/imagenet_frlc_r<rank>_n<n>_c<n_classes>.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import PATHS, add_code_dirs
add_code_dirs()
from run_omt import (EMB_PATH, OUT_DIR, RESNET_PATH, load_classifier,
                          load_source_target, nearest_target, predict, topk_hit)
from FRLC import FRLC_LR_opt
from HR_OT import compute_lr_sqeuclidean_matrix


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
    p.add_argument("--rank", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=90)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--min-iter", type=int, default=15)
    p.add_argument("--tau-in", type=float, default=100000)
    p.add_argument("--max-inneriters", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--emb", default=EMB_PATH)
    p.add_argument("--device", default=None, help="GPU for classifier / NN search")
    p.add_argument("--solve-device", default="cpu", help="device for the FRLC solve")
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
    sdev = torch.device(args.solve_device)
    torch.set_num_threads(os.cpu_count())
    # X, Y stay in CPU RAM; only the cost factors and Q/R go to the solve device.
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    Y_t = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
    clf = load_classifier(RESNET_PATH, d, device)

    C_factors = tuple(c.to(sdev) for c in
                      compute_lr_sqeuclidean_matrix(X_t, Y_t, rescale_cost=True))
    print(f"Running FRLC rank {args.rank} on n={n} ...", flush=True)
    t0 = time.time()
    Q, R, T, errs = FRLC_LR_opt(
        C_factors, None, None, gamma=args.gamma, r=args.rank,
        max_iter=args.max_iter, min_iter=args.min_iter, device=sdev,
        max_inneriters_balanced=args.max_inneriters,
        max_inneriters_relaxed=args.max_inneriters,
        diagonalize_return=True, printCost=False, tau_in=args.tau_in,
        dtype=torch.float32)
    solve_time = time.time() - t0
    print(f"  solved in {solve_time:.1f}s", flush=True)
    del C_factors
    Q, R, T = Q.cpu(), R.cpu(), T.cpu()               # post-processing on CPU
    sdev = torch.device("cpu")
    g_inv = 1.0 / torch.diag(T)                       # P = Q diag(1/g) R^T
    Qs = Q * g_inv[None, :]
    a = Q.sum(1, keepdim=True)                        # row marginal (~1/n)
    X_hat = (Qs @ (R.T @ Y_t)) / a

    # <P, C> with squared Euclidean C, via ||x||^2 + ||y||^2 - 2 x.y in factored form
    xx, yy = (X_t ** 2).sum(1), (Y_t ** 2).sum(1)
    b = R.sum(1)
    plan_cost = float((a.squeeze() * xx).sum() + (b * yy).sum()
                      - 2 * ((Qs.T @ X_t) * (R.T @ Y_t)).sum())

    # row-argmax of P and plan-weighted class accuracy (chunked)
    match = np.empty(n, dtype=np.int64)
    RT = R.T.contiguous()
    cY = torch.as_tensor(class_Y, device=sdev)
    # class mass of each latent factor: R^T 1_{class}
    RclassT = torch.zeros(int(max(class_X.max(), class_Y.max())) + 1, args.rank, device=sdev).index_add_(0, cY, R)
    cX = torch.as_tensor(class_X, device=sdev)
    plan_class_acc = float((Qs * RclassT[cX]).sum())
    for s in range(0, n, 1024):
        Pc = Qs[s:s + 1024] @ RT
        match[s:s + 1024] = Pc.argmax(1).cpu().numpy()
    del RT

    displacement = float(torch.norm(X_t - X_hat, dim=1).mean())
    nn_idx, nn_sq = nearest_target_gpu(X_hat, Y_t, device)
    pred_X, top5_X = predict(clf, X, device)
    pred_Y, _ = predict(clf, Y, device)
    pred_hat, top5_hat = predict(clf, X_hat.cpu().numpy(), device)

    m = {
        "method": "frlc", "rank": args.rank, "gamma": args.gamma,
        "max_iter": args.max_iter, "min_iter": args.min_iter, "tau_in": args.tau_in,
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
        "solve_time_s": solve_time,
    }
    print(json.dumps(m, indent=2))
    out = args.out or os.path.join(
        OUT_DIR, f"imagenet_frlc_r{args.rank}_n{n}_c{n_classes}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Metrics written to {out}", flush=True)


if __name__ == "__main__":
    main()
