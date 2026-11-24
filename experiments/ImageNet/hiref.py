#!/usr/bin/env python3
"""
HiRef (Hierarchical Refinement OT) alignment of ImageNet ResNet50 embeddings,
with the same class-preservation evaluation used in run_omt.py.

Method
------
Exactly the recipe from experiments/ImageNet._Hiref.ipynb:

    rank_schedule = rank_annealing.optimal_rank_schedule(
        n, hierarchy_depth=6, max_Q=int(2**11), max_rank=64)
    hrot = HR_OT.HierarchicalRefinementOT.init_from_point_clouds(
        _X, _Y, rank_schedule, base_rank=1, device=device)
    F = hrot.run(return_as_coupling=False)
    cost = hrot.compute_OT_cost()

with the notebook's float32 point clouds and its default cost (Euclidean, i.e.
`sq_Euclidean=False`, the notebook's p=1) and default FRLC solver params.
HiRef returns a set of Monge pairs (i, j) -- a full 1-1 alignment -- so the
transport map is T(x_i) = Y[j], landing on an actual target image.

Data & evaluation are shared with run_omt.py (imported from it), so the
numbers are directly comparable to the sinkhorn / emd / cvx / gmmot runs:
1. ResNet50 embeddings from <embeddings> (2048-d).
2. Class-stratified random source/target split: each class contributes the same
   number of distinct images to X as to Y.
3. HiRef alignment in the latent space.
4. Transport cost *and* class preservation -- matched-pair class accuracy plus
   the frozen ResNet50 `fc` head applied to the transported samples.

Note on sizes: the rank schedule must factorise n exactly, and n must not be
prime.  optimal_rank_schedule() asserts this; if it fails for your n, adjust
--per-class / --n-classes (e.g. 1000, 2000, 4000, 5000, 10000 all work).

Usage
-----
    python hiref.py --per-class 2                    # 2000 per side
    python hiref.py --per-class 1                    # 1000 per side
    python hiref.py --n-classes 100 --per-class 2    # 200 per side

Output
------
    <results_dir>/imagenet_hiref_n<n>_c<n_classes>.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import PATHS, add_code_dirs
add_code_dirs()

# Shared data loading / classifier / metrics -- identical to the OMT experiment.
from run_omt import (EMB_PATH, OUT_DIR, RESNET_PATH, load_classifier,
                          load_source_target, nearest_target, predict, topk_hit)

# HiRef, as imported in the notebook.
import rank_annealing
import HR_OT


# ── HiRef ────────────────────────────────────────────────────────────────────
def solve_hiref(X_t, Y_t, hierarchy_depth, max_Q, max_rank, base_rank,
                sq_euclidean, device):
    """Run Hierarchical Refinement exactly as experiments/ImageNet._Hiref.ipynb does.

    Returns (match, cost, info) where `match[i]` is the index in Y that HiRef
    pairs with X[i] -- HiRef produces a 1-1 alignment, so this is a permutation.
    """
    n = X_t.shape[0]
    rank_schedule = rank_annealing.optimal_rank_schedule(
        n, hierarchy_depth=hierarchy_depth, max_Q=max_Q, max_rank=max_rank)
    print(f"rank schedule: {rank_schedule} (product = "
          f"{int(np.prod(rank_schedule))}, n = {n})", flush=True)

    print("Running HiRef hierarchical refinement ...", flush=True)
    t0 = time.time()
    hrot = HR_OT.HierarchicalRefinementOT.init_from_point_clouds(
        X_t, Y_t, rank_schedule, base_rank=base_rank, device=device,
        sq_Euclidean=sq_euclidean)
    F = hrot.run(return_as_coupling=False)
    solve_time = time.time() - t0
    cost = float(hrot.compute_OT_cost())
    print(f"  solved in {solve_time:.1f}s | HiRef OT cost: {cost:.6f}", flush=True)

    # F is a list of (idxX, idxY) singleton co-clusters at base_rank=1.
    match = np.empty(n, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    for idx_x, idx_y in F:
        i = int(np.asarray(idx_x.cpu()).reshape(-1)[0])
        j = int(np.asarray(idx_y.cpu()).reshape(-1)[0])
        match[i] = j
        seen[i] = True
    if not seen.all():
        raise RuntimeError(f"HiRef left {int((~seen).sum())} source points unmatched")

    info = {"solve_time_s": solve_time,
            "rank_schedule": [int(r) for r in rank_schedule],
            "hierarchy_depth": hierarchy_depth,
            "max_Q": int(max_Q),
            "max_rank": int(max_rank),
            "base_rank": int(base_rank),
            "sq_Euclidean": bool(sq_euclidean),
            "n_monge_pairs": len(F),
            "hiref_ot_cost": cost}
    return match, cost, info


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-pairs", type=int, default=10000,
                        help="approximate samples per side; rounded down to a whole "
                             "number of images per class (n_pairs // n_classes)")
    parser.add_argument("--per-class", type=int, default=None,
                        help="images per class per side; overrides --n-pairs")
    parser.add_argument("--n-classes", type=int, default=None,
                        help="use a random subset of this many classes (keeps the "
                             "split stratified)")
    # HiRef hyperparameters -- defaults are the notebook's.
    parser.add_argument("--hierarchy-depth", type=int, default=6)
    parser.add_argument("--max-q", type=int, default=int(2 ** 11))
    parser.add_argument("--max-rank", type=int, default=64)
    parser.add_argument("--base-rank", type=int, default=1)
    parser.add_argument("--sq-euclidean", action="store_true",
                        help="squared-Euclidean cost (notebook default is Euclidean)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emb", default=EMB_PATH)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--out", default=None, help="path for the JSON report")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"On device: {device}", flush=True)
    torch.manual_seed(args.seed)

    # 1-2. Embeddings, class-stratified random split into source / target.
    X, class_X, Y, class_Y = load_source_target(args.emb, args.n_pairs, args.seed,
                                                per_class=args.per_class,
                                                n_classes_subset=args.n_classes)
    n_side = X.shape[0]
    n_classes = len(np.unique(class_X))
    per_class = n_side // n_classes

    # The notebook builds float32 point clouds on the GPU.
    X_t = torch.tensor(np.array(X)).float().to(device)
    Y_t = torch.tensor(np.array(Y)).float().to(device)

    # ResNet50 classifier head, used to label the transported samples.
    clf = load_classifier(RESNET_PATH, X.shape[1], device)

    # 3. HiRef alignment.
    match, hiref_cost, ot_info = solve_hiref(
        X_t, Y_t, args.hierarchy_depth, args.max_q, args.max_rank,
        args.base_rank, args.sq_euclidean, device)

    # HiRef gives a 1-1 map onto real target points: T(x_i) = Y[match[i]].
    X_hat = Y_t[torch.as_tensor(match, device=device)]

    # Same cost summaries as the OMT script, so the two are comparable.
    # <P, C> for the HiRef permutation, with C = squared Euclidean (as in
    # run_omt.py's transport_cost_plan) -- P puts mass 1/n on each pair.
    sq_pair = torch.norm(X_t - X_hat, dim=1) ** 2
    transport_cost = float(sq_pair.mean())
    displacement = float(torch.norm(X_t - X_hat, dim=1).mean())
    nn_idx, nn_sq = nearest_target(X_hat, Y_t)
    nn_dist = float(np.sqrt(nn_sq).mean())

    # 4a. Class preservation of the matching.
    matched_class_acc = float(np.mean(class_Y[match] == class_X))
    nn_class_acc = float(np.mean(class_Y[nn_idx] == class_X))
    rng = np.random.default_rng(args.seed + 1)
    chance_acc = float(np.mean(class_Y[rng.permutation(len(class_Y))] == class_X))

    # 4b. Class preservation under the ResNet50 classifier head.
    X_hat_np = X_hat.cpu().numpy()
    pred_X, top5_X = predict(clf, X, device)
    pred_Y, _ = predict(clf, Y, device)
    pred_hat, top5_hat = predict(clf, X_hat_np, device)

    metrics = {
        "method": "hiref",
        "n_per_side": int(n_side),
        "n_classes": int(n_classes),
        "per_class_per_side": int(per_class),
        "dim": int(X.shape[1]),
        "seed": args.seed,
        # ── transport cost ──
        "transport_cost_plan": transport_cost,                # <P, C>, C squared Euclidean
        "hiref_ot_cost": hiref_cost,                          # compute_OT_cost() (Euclidean)
        "mean_displacement_L2": displacement,                 # ||x - T(x)||
        "mean_displacement_L2_per_dim": displacement / X.shape[1],
        "mean_dist_to_nearest_target": nn_dist,               # ||T(x) - y_NN||
        # ── class preservation of the alignment ──
        "matched_class_accuracy": matched_class_acc,          # class(x) == class(matched y)
        "nearest_neighbour_class_accuracy": nn_class_acc,
        "plan_weighted_class_accuracy": matched_class_acc,    # 1-1 map -> same as matched
        "random_matching_class_accuracy": chance_acc,         # chance baseline
        # ── ResNet50 classifier on the transported points ──
        "clf_top1_source": float(np.mean(pred_X == class_X)),
        "clf_top1_target": float(np.mean(pred_Y == class_Y)),
        "clf_top1_transported": float(np.mean(pred_hat == class_X)),
        "clf_top5_source": topk_hit(top5_X, class_X),
        "clf_top5_transported": topk_hit(top5_hat, class_X),
        "label_preservation_top1": float(np.mean(pred_hat == pred_X)),
        "label_preservation_top5": topk_hit(top5_hat, pred_X),
        "clf_agreement_with_matched_target": float(np.mean(pred_hat == pred_Y[match])),
        "n_distinct_targets_matched": int(len(np.unique(match))),
    }
    metrics.update({f"ot_{k}": v for k, v in ot_info.items()})

    print("\n" + "=" * 68)
    print(f"HiRef alignment of ImageNet embeddings  |  "
          f"n={n_side} per side ({per_class}/class x {n_classes} classes)")
    print("=" * 68)
    print("-- transport cost")
    print(f"  HiRef compute_OT_cost()             : {hiref_cost:.6f}")
    print(f"  <P, C> (squared Euclidean)          : {transport_cost:.4f}")
    print(f"  mean ||x - T(x)||                   : {displacement:.4f} "
          f"({displacement / X.shape[1]:.3e} per dim)")
    print(f"  mean ||T(x) - nearest y||           : {nn_dist:.4f}")
    print("-- class preservation of the matching")
    print(f"  matched-pair class accuracy         : {matched_class_acc:.4f}")
    print(f"  nearest-neighbour class accuracy    : {nn_class_acc:.4f}")
    print(f"  random-matching baseline (chance)    : {chance_acc:.4f}")
    print("-- ResNet50 classifier head on transported samples")
    print(f"  top-1 on source X                   : {metrics['clf_top1_source']:.4f}")
    print(f"  top-1 on target Y                   : {metrics['clf_top1_target']:.4f}")
    print(f"  top-1 on T(X) vs true source class  : {metrics['clf_top1_transported']:.4f}")
    print(f"  top-5 on T(X) vs true source class  : {metrics['clf_top5_transported']:.4f}")
    print(f"  label preservation clf(T(x))==clf(x): {metrics['label_preservation_top1']:.4f}")
    print(f"  clf(T(x)) == clf(matched y)         : "
          f"{metrics['clf_agreement_with_matched_target']:.4f}")
    print(f"  distinct target images matched      : "
          f"{metrics['n_distinct_targets_matched']} / {n_side}")
    print("=" * 68 + "\n")

    out_path = args.out or os.path.join(
        OUT_DIR, f"imagenet_hiref_n{n_side}_c{n_classes}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
