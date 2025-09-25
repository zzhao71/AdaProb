# tsne_group_inplot_labels_with_kl.py
# Square t-SNE panel (colored by group) with a small in-plot legend (top-right)
# and console output of pairwise KL (both directions) + JS divergences.

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------- IO helpers ----------
def load_2d_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        t = obj
    elif isinstance(obj, (list, tuple)):
        t = next((it for it in obj if torch.is_tensor(it) and it.ndim == 2), None)
        if t is None:
            raise ValueError(f"{path}: no 2D tensor found in the saved object.")
    else:
        raise TypeError(f"{path}: expected tensor or list/tuple; got {type(obj)}")
    if t.ndim != 2:
        raise ValueError(f"{path}: expected shape [N, K], got {tuple(t.shape)}")
    return t.float().cpu()

def to_probs(x: torch.Tensor, assume_logits: bool) -> torch.Tensor:
    if assume_logits:
        return torch.softmax(x, dim=1)
    s = x.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (x / s).clamp_min(0.0)

# ---------- Divergence helpers (centroid-based) ----------
EPS = 1e-8

def _norm(v):
    v = v.astype(np.float64) + EPS
    v /= v.sum(axis=-1, keepdims=True)
    return v

def kl(p, q):
    p = _norm(p); q = _norm(q)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)

def js(p, q):
    p = _norm(p); q = _norm(q)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def centroid_vec(X):  # mean probability vector
    v = X.mean(axis=0, keepdims=True)
    return _norm(v)[0]

# ---------- Main ----------
def main(args):
    out_dir = Path(args.out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)

    # Load & convert
    A = to_probs(load_2d_tensor(args.path_test_logits),  True).numpy()
    B = to_probs(load_2d_tensor(args.path_scrub_logits), True).numpy()
    C = to_probs(load_2d_tensor(args.path_test_probs),   False).numpy()
    X = np.vstack([A, B, C]).astype(np.float64)
    group = np.array([0]*len(A) + [1]*len(B) + [2]*len(C))

    # PCA -> t-SNE (joint)
    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=min(50, Xs.shape[1]), random_state=args.seed).fit_transform(Xs)
    perplexity = min(args.perplexity, max(5, Xp.shape[0] // 4))
    Z = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto",
             init="pca", random_state=args.seed, n_iter=args.n_iter,
             metric="euclidean", verbose=1).fit_transform(Xp)

    # Plot (square, title, small legend in top-right)
    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.labelsize": args.font_size,
        "xtick.labelsize": args.font_size-1,
        "ytick.labelsize": args.font_size-1,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(args.size, args.size))
    ax.set_title(f"Lacuna-5 (ALLCNN)", pad=10)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=group, s=args.marker_size, alpha=1.0, linewidths=0)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    # square data window
    x_min, x_max = Z[:,0].min(), Z[:,0].max()
    y_min, y_max = Z[:,1].min(), Z[:,1].max()
    xc, yc = 0.5*(x_min+x_max), 0.5*(y_min+y_max)
    half = 0.5*max(x_max-x_min, y_max-y_min)*1.05
    ax.set_xlim(xc-half, xc+half); ax.set_ylim(yc-half, yc+half)
    ax.set_aspect("equal", adjustable="box")

    # small legend INSIDE top-right
    names = ["AdaProb", "SCRUB", "Retrain"]
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=sc.cmap(sc.norm(i)),
                           label=names[i], markersize=max(5, int(np.sqrt(args.marker_size))))
               for i in range(3)]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True, fancybox=True,
                    borderpad=0.25, handletextpad=0.4, labelspacing=0.3,
                    borderaxespad=0.6, prop={"size": args.legend_font_size})
    leg.get_frame().set_alpha(0.85); leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    out_path = out_dir / "tsne_group_inplot_labels.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"[OK] saved: {out_path}")

    # ---- Pairwise divergences (centroid-based) ----
    Pa, Pb, Pc = centroid_vec(A), centroid_vec(B), centroid_vec(C)
    pairs = [
        ("A(Test log→prob) vs B(Scrub log→prob)", Pa, Pb),
        ("A(Test log→prob) vs C(Test probs)",      Pa, Pc),
        ("B(Scrub log→prob) vs C(Test probs)",     Pb, Pc),
    ]
    lines = []
    for name, P, Q in pairs:
        kl_pq = float(kl(P[None, :], Q[None, :]))
        kl_qp = float(kl(Q[None, :], P[None, :]))
        js_pq = float(js(P[None, :], Q[None, :]))
        line = f"{name}: KL(P||Q)={kl_pq:.4f}  KL(Q||P)={kl_qp:.4f}  JS={js_pq:.4f}"
        lines.append(line)

    print("\nPairwise divergences (centroid-based):")
    for s in lines:
        print("  " + s)
    (out_dir / "divergences.txt").write_text("\n".join(lines))
    print(f"[OK] divergences written to: {out_dir/'divergences.txt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_test_logits",  type=str, default="/mnt/data/test_lacuna5_resnet_final_all_outputs_tensor.pt")
    parser.add_argument("--path_scrub_logits", type=str, default="/mnt/data/scrub_test_lacuna5_resnet_final_all_outputs_tensor.pt")
    parser.add_argument("--path_test_probs",   type=str, default="/mnt/data/test_lacuna5_resnet_final_all_probs_tensor.pt")
    parser.add_argument("--out_dir",           type=str, default="./tsne_outputs")
    parser.add_argument("--n_iter",            type=int, default=650)
    parser.add_argument("--perplexity",        type=int, default=30)
    parser.add_argument("--seed",              type=int, default=42)
    # presentation controls
    parser.add_argument("--size",            type=float, default=8.0)
    parser.add_argument("--marker_size",     type=float, default=18.0)
    parser.add_argument("--font_size",       type=int,   default=16)
    parser.add_argument("--legend_font_size",type=int,   default=10)
    args = parser.parse_args()
    main(args)
