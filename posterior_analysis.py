"""
posterior_analysis.py
=====================
Single entry-point for all post-training analysis of the archaeological LiDAR
contrastive-learning pipeline.

What it does, in order:
  1.  Dense-scan every raster tile → embeddings + UTM coordinates for every patch.
  2.  Train a small GPU MLP classifier on top of the frozen encoder
        positives  = patches centred on known sites
        negatives  = random background patches (buffered away from sites)
  3.  Score every patch → site_likelihood  (0–1, sigmoid output).
  4.  Plot 1 – UMAP embedding coloured by likelihood, known sites as stars.
  5.  Plot 2 – Real geographic map (UTM easting/northing), colour = likelihood,
                known sites overlaid as stars.
  6.  Write  <output_dir>/site_likelihood.csv   — one row per patch:
        easting, northing, patch_cx, patch_cy, tile, site_likelihood
        (same spatial grain as the input scan, just with an extra column)

Usage example (on the cluster):
    python posterior_analysis.py \
        --data_path /cluster/medbow/project/galacticbulge/ruin_repo \
        --model_dir models \
        --output_dir posterior_results \
        --patch_size 256 \
        --stride 128 \
        --negatives_per_raster 3000 \
        --mlp_epochs 50 \
        --batch_size 512
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from shapely.geometry import Point, box
from shapely.ops import unary_union
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import umap

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Local imports (same directory as this script)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import sample_site_patches, sample_random_patches, get_raster_paths
from model import load_models
from visualization import compute_embeddings


# ===========================================================================
# 1.  DENSE SCAN  ─  slide window over every tile, record coords + embeddings
# ===========================================================================

def dense_scan_all_tiles(encoder, raster_paths, patch_size, stride, batch_size, device):
    """
    Slide a window over every raster and return embeddings + centre coordinates.

    Returns
    -------
    embeddings  : np.ndarray  (N, D)
    eastings    : np.ndarray  (N,)   UTM easting  of patch centre
    northings   : np.ndarray  (N,)   UTM northing of patch centre
    tile_names  : list[str]   (N,)
    raster_crs  : str  – CRS of the last raster (all tiles should share CRS)
    """
    all_embeddings, all_eastings, all_northings, all_tiles = [], [], [], []
    raster_crs = None

    encoder.eval()
    encoder.to(device)

    for raster_path in tqdm(raster_paths, desc="Dense scan tiles"):
        tile_name = os.path.basename(raster_path)
        try:
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs.to_string()
                h, w = src.height, src.width
                nodata = src.nodata

                batch_patches, batch_east, batch_north = [], [], []

                def flush_batch():
                    if not batch_patches:
                        return
                    arr = np.array(batch_patches, dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr[:, np.newaxis, :, :]          # N,1,H,W
                    elif arr.ndim == 4 and arr.shape[-1] in {1, 3}:
                        arr = arr.transpose(0, 3, 1, 2)
                    t = torch.from_numpy(arr).to(device)
                    with torch.no_grad():
                        emb = encoder(t).cpu().numpy()
                    all_embeddings.append(emb)
                    all_eastings.extend(batch_east)
                    all_northings.extend(batch_north)
                    all_tiles.extend([tile_name] * len(batch_patches))
                    batch_patches.clear(); batch_east.clear(); batch_north.clear()

                for row in range(0, h - patch_size + 1, stride):
                    for col in range(0, w - patch_size + 1, stride):
                        window = ((row, row + patch_size), (col, col + patch_size))
                        patch = src.read(window=window)

                        if nodata is not None and np.any(patch == nodata):
                            continue
                        if np.all(patch == 0):
                            continue

                        if patch.shape[0] == 1:
                            patch = patch[0]        # H,W

                        # Normalise to [0,1]
                        pmin, pmax = patch.min(), patch.max()
                        if pmax > pmin:
                            patch = (patch - pmin) / (pmax - pmin)

                        # Centre of this patch in UTM
                        cx_px = col + patch_size / 2
                        cy_px = row + patch_size / 2
                        east, north = src.transform * (cx_px, cy_px)

                        batch_patches.append(patch)
                        batch_east.append(east)
                        batch_north.append(north)

                        if len(batch_patches) >= batch_size:
                            flush_batch()

                flush_batch()

        except Exception as exc:
            print(f"  Warning: skipping {tile_name}: {exc}")
            continue

    if not all_embeddings:
        raise RuntimeError("No patches extracted during dense scan. Check raster paths.")

    embeddings  = np.concatenate(all_embeddings, axis=0)
    eastings    = np.array(all_eastings,  dtype=np.float64)
    northings   = np.array(all_northings, dtype=np.float64)

    print(f"Dense scan complete: {len(embeddings):,} patches from {len(raster_paths)} tiles")
    return embeddings, eastings, northings, all_tiles, raster_crs


# ===========================================================================
# 2.  COLLECT LABELLED EMBEDDINGS FOR MLP TRAINING
# ===========================================================================

def load_sites_gdf(data_path, crs="EPSG:26912"):
    csv_candidates = [
        os.path.join(data_path, "labeled_data", "site_locations.csv"),
        os.path.join(data_path, "site_locations", "site_locations_mvnp.csv"),
        os.path.join(data_path, "site_locations.csv"),
    ]
    for path in csv_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if {"true_easting", "true_northing"}.issubset(df.columns):
                geom = [Point(x, y) for x, y in zip(df["true_easting"], df["true_northing"])]
                gdf = gpd.GeoDataFrame(df, geometry=geom, crs=crs)
                print(f"Loaded {len(gdf)} known sites from {path}")
                return gdf
            elif {"longitude", "latitude"}.issubset(df.columns):
                geom = [Point(x, y) for x, y in zip(df["longitude"], df["latitude"])]
                gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
                print(f"Loaded {len(gdf)} known sites from {path}")
                return gdf
    raise FileNotFoundError(f"Could not find site CSV in {data_path}")


def collect_labelled_embeddings(encoder, raster_paths, sites_gdf,
                                 patch_size, negatives_per_raster,
                                 buffer_m, batch_size, device):
    """Return (positive_embeddings, negative_embeddings)."""
    encoder.eval()
    encoder.to(device)

    pos_embs, neg_embs = [], []

    for raster_path in tqdm(raster_paths, desc="Collecting labelled embeddings"):
        try:
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
            aligned = sites_gdf.to_crs(raster_crs) if sites_gdf.crs != raster_crs else sites_gdf

            # --- positives ---
            site_patches, _, _ = sample_site_patches([raster_path], aligned, patch_size=patch_size)
            if len(site_patches) > 0:
                emb = compute_embeddings(encoder, site_patches, batch_size=batch_size)
                pos_embs.append(emb)

            # --- negatives (buffered away from sites) ---
            bg_patches, bg_locs, _ = sample_random_patches(
                [raster_path], patch_size=patch_size, n_samples=negatives_per_raster
            )
            if len(bg_patches) > 0 and buffer_m > 0 and not aligned.empty:
                buffered = unary_union(aligned.geometry.buffer(buffer_m))
                keep = [i for i, loc in enumerate(bg_locs)
                        if not box(*loc).intersects(buffered)]
                bg_patches = bg_patches[keep]
            if len(bg_patches) > 0:
                emb = compute_embeddings(encoder, bg_patches, batch_size=batch_size)
                neg_embs.append(emb)

        except Exception as exc:
            print(f"  Warning: skipping {raster_path}: {exc}")

    pos = np.vstack(pos_embs) if pos_embs else np.empty((0, 1))
    neg = np.vstack(neg_embs) if neg_embs else np.empty((0, 1))
    print(f"Labelled embeddings: {len(pos)} positives, {len(neg)} negatives")
    return pos, neg


# ===========================================================================
# 3.  MLP SITE CLASSIFIER  (trained on GPU)
# ===========================================================================

class SiteClassifierMLP(nn.Module):
    """Small MLP: embedding_dim → 256 → 128 → 64 → 1 (sigmoid)."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),    nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp_classifier(pos_embs, neg_embs, device, epochs, batch_size, lr=1e-3):
    """
    Train the MLP on GPU.  Returns trained model + val AUC.
    """
    X = np.vstack([pos_embs, neg_embs]).astype(np.float32)
    y = np.concatenate([np.ones(len(pos_embs)), np.zeros(len(neg_embs))]).astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    ds_tr  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dl_tr  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SiteClassifierMLP(X_tr.shape[1]).to(device)

    # Class-weighted BCE to handle imbalance
    pos_weight = torch.tensor([len(neg_embs) / max(len(pos_embs), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc, best_state = 0.0, None
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        train_losses.append(epoch_loss / len(ds_tr))

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                probs = torch.sigmoid(model(xb.to(device))).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(yb.numpy())
        auc = roc_auc_score(all_labels, all_probs)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs}  loss={train_losses[-1]:.4f}  val_AUC={auc:.4f}")

    model.load_state_dict(best_state)
    print(f"\nMLP training done.  Best val AUC = {best_auc:.4f}")

    # Full val report
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dl_val:
            probs = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            all_probs.extend(probs); all_labels.extend(yb.numpy())
    preds = (np.array(all_probs) >= 0.5).astype(int)
    print(classification_report(all_labels, preds, target_names=["background", "site"]))

    return model, best_auc, train_losses


def score_embeddings(mlp, embeddings, batch_size, device):
    """Run MLP on all embeddings → site_likelihood [0,1]."""
    mlp.eval().to(device)
    scores = []
    for i in range(0, len(embeddings), batch_size):
        chunk = torch.from_numpy(embeddings[i:i+batch_size].astype(np.float32)).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(mlp(chunk)).cpu().numpy()
        scores.extend(probs)
    return np.array(scores, dtype=np.float32)


# ===========================================================================
# 4.  PLOTS
# ===========================================================================

def plot_embedding_space(embeddings, scores, site_embs, output_dir):
    """
    UMAP coloured by site_likelihood; known site embeddings shown as stars.
    """
    print("Computing UMAP (this can take a few minutes for large scans)…")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05,
                        random_state=42, n_jobs=-1, verbose=False)

    # Fit on a representative subsample to keep it tractable, then transform all
    max_fit = 50_000
    if len(embeddings) > max_fit:
        idx = np.random.choice(len(embeddings), max_fit, replace=False)
        reducer.fit(embeddings[idx])
    else:
        reducer.fit(embeddings)

    all_embs = embeddings
    site_proj = None
    if len(site_embs) > 0:
        all_embs = np.vstack([embeddings, site_embs])

    proj = reducer.transform(all_embs)
    bg_proj   = proj[:len(embeddings)]
    if len(site_embs) > 0:
        site_proj = proj[len(embeddings):]

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(
        bg_proj[:, 0], bg_proj[:, 1],
        c=scores, cmap="RdYlGn", s=4, alpha=0.5, vmin=0, vmax=1,
        rasterized=True
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03)
    cbar.set_label("Site likelihood", fontsize=12)

    if site_proj is not None:
        ax.scatter(
            site_proj[:, 0], site_proj[:, 1],
            c="blue", marker="*", s=200, zorder=5,
            edgecolors="white", linewidths=0.5,
            label="Known sites"
        )
        ax.legend(fontsize=11)

    ax.set_title("Latent Space  –  coloured by site likelihood", fontsize=14)
    ax.set_xlabel("UMAP dim 1"); ax.set_ylabel("UMAP dim 2")
    ax.set_facecolor("#1a1a1a")
    fig.tight_layout()
    out = os.path.join(output_dir, "embedding_space.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding plot  → {out}")


def plot_geographic_map(eastings, northings, scores, sites_gdf, raster_crs, output_dir):
    """
    Real UTM geographic map; colour = site_likelihood; known sites as stars.
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    sc = ax.scatter(
        eastings, northings,
        c=scores, cmap="RdYlGn", s=3, alpha=0.6, vmin=0, vmax=1,
        rasterized=True
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03)
    cbar.set_label("Site likelihood", fontsize=12)

    # Overlay known sites in the correct CRS
    if sites_gdf is not None and len(sites_gdf) > 0:
        try:
            s = sites_gdf.to_crs(raster_crs) if sites_gdf.crs.to_string() != raster_crs else sites_gdf
            s_east  = s.geometry.x.values
            s_north = s.geometry.y.values
            ax.scatter(
                s_east, s_north,
                c="blue", marker="*", s=250, zorder=6,
                edgecolors="white", linewidths=0.4,
                label="Known sites"
            )
            ax.legend(fontsize=11)
        except Exception as e:
            print(f"  Warning: could not overlay known sites on map: {e}")

    ax.set_title("Archaeological site likelihood across survey area", fontsize=14)
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.set_facecolor("#1a1a1a")
    fig.tight_layout()
    out = os.path.join(output_dir, "geographic_likelihood_map.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved geographic map   → {out}")


def plot_training_curve(train_losses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, color="steelblue")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
    ax.set_title("MLP classifier – training loss")
    fig.tight_layout()
    out = os.path.join(output_dir, "mlp_training_loss.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved MLP training curve → {out}")


# ===========================================================================
# 5.  MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Posterior analysis – site likelihood mapping")
    p.add_argument("--data_path",            type=str,   default="/cluster/medbow/project/galacticbulge/ruin_repo")
    p.add_argument("--model_dir",            type=str,   default="models",
                   help="Relative to data_path or absolute")
    p.add_argument("--output_dir",           type=str,   default="posterior_results")
    p.add_argument("--patch_size",           type=int,   default=256)
    p.add_argument("--stride",               type=int,   default=128,
                   help="Stride for the dense scan (smaller = more patches, slower)")
    p.add_argument("--raster_limit",         type=int,   default=None,
                   help="Limit number of rasters (None = all)")
    p.add_argument("--negatives_per_raster", type=int,   default=3000)
    p.add_argument("--negative_buffer_m",    type=float, default=30.0,
                   help="Buffer (metres) around known sites to exclude negatives")
    p.add_argument("--mlp_epochs",           type=int,   default=50)
    p.add_argument("--mlp_lr",               type=float, default=1e-3)
    p.add_argument("--batch_size",           type=int,   default=512)
    p.add_argument("--sites_crs",            type=str,   default="EPSG:26912")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs via DataParallel")

    # ── resolve model dir ──────────────────────────────────────────────────
    model_dir = args.model_dir if os.path.isabs(args.model_dir) \
                else os.path.join(args.data_path, args.model_dir)

    print("\n[1/6] Loading encoder…")
    encoder, _ = load_models(save_dir=model_dir, device=device)
    encoder.eval()
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
    encoder.to(device)

    # ── raster paths ───────────────────────────────────────────────────────
    raster_paths = get_raster_paths(args.data_path, limit=args.raster_limit)
    print(f"   Found {len(raster_paths)} raster tiles")

    # ── known sites ────────────────────────────────────────────────────────
    print("\n[2/6] Loading known site locations…")
    sites_gdf = load_sites_gdf(args.data_path, crs=args.sites_crs)

    # ── dense scan ─────────────────────────────────────────────────────────
    print(f"\n[3/6] Dense scan (patch={args.patch_size}, stride={args.stride})…")
    scan_embs, eastings, northings, tile_names, raster_crs = dense_scan_all_tiles(
        encoder      = encoder,
        raster_paths = raster_paths,
        patch_size   = args.patch_size,
        stride       = args.stride,
        batch_size   = args.batch_size,
        device       = device,
    )

    # ── collect labelled embeddings for MLP training ───────────────────────
    print(f"\n[4/6] Collecting labelled embeddings for MLP training…")
    pos_embs, neg_embs = collect_labelled_embeddings(
        encoder              = encoder,
        raster_paths         = raster_paths,
        sites_gdf            = sites_gdf,
        patch_size           = args.patch_size,
        negatives_per_raster = args.negatives_per_raster,
        buffer_m             = args.negative_buffer_m,
        batch_size           = args.batch_size,
        device               = device,
    )

    # Also keep a copy of site embeddings for the UMAP overlay
    site_embs_for_plot = pos_embs.copy()

    # ── train MLP ─────────────────────────────────────────────────────────
    print(f"\n[5/6] Training MLP site classifier ({args.mlp_epochs} epochs, device={device})…")
    mlp, best_auc, train_losses = train_mlp_classifier(
        pos_embs   = pos_embs,
        neg_embs   = neg_embs,
        device     = device,
        epochs     = args.mlp_epochs,
        batch_size = args.batch_size,
        lr         = args.mlp_lr,
    )
    torch.save(mlp.state_dict(), os.path.join(args.output_dir, "site_classifier_mlp.pt"))
    print(f"   Saved MLP weights → {args.output_dir}/site_classifier_mlp.pt")

    # ── score every scanned patch ─────────────────────────────────────────
    print(f"\n[6/6] Scoring {len(scan_embs):,} patches…")
    # Use raw (not DataParallel) encoder's embedding dim
    _enc = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
    scores = score_embeddings(mlp, scan_embs, args.batch_size, device)

    # ── plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_training_curve(train_losses, args.output_dir)

    plot_embedding_space(
        embeddings = scan_embs,
        scores     = scores,
        site_embs  = site_embs_for_plot,
        output_dir = args.output_dir,
    )

    plot_geographic_map(
        eastings   = eastings,
        northings  = northings,
        scores     = scores,
        sites_gdf  = sites_gdf,
        raster_crs = raster_crs,
        output_dir = args.output_dir,
    )

    # ── output CSV ─────────────────────────────────────────────────────────
    print("\nWriting output CSV…")
    df_out = pd.DataFrame({
        "easting":          eastings,
        "northing":         northings,
        "tile":             tile_names,
        "site_likelihood":  scores,
    })

    # Mark which patches coincide with a known site (useful ground-truth column)
    try:
        pts = gpd.GeoDataFrame(
            df_out,
            geometry=gpd.points_from_xy(eastings, northings),
            crs=raster_crs,
        )
        s_aligned = sites_gdf.to_crs(raster_crs)
        s_buffered = s_aligned.geometry.buffer(args.patch_size / 2)
        s_union = unary_union(s_buffered)
        df_out["is_known_site"] = pts.geometry.within(s_union).astype(int)
    except Exception as e:
        print(f"  (Could not add is_known_site column: {e})")
        df_out["is_known_site"] = np.nan

    csv_path = os.path.join(args.output_dir, "site_likelihood.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV ({len(df_out):,} rows) → {csv_path}")

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("POSTERIOR ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Tiles scanned          : {len(set(tile_names))}")
    print(f"  Total patches scored   : {len(scores):,}")
    print(f"  MLP best val AUC       : {best_auc:.4f}")
    print(f"  High-confidence (≥0.7) : {(scores >= 0.7).sum():,} patches")
    print(f"  Output directory       : {os.path.abspath(args.output_dir)}")
    print("\n  Files written:")
    for f in ["embedding_space.png", "geographic_likelihood_map.png",
              "mlp_training_loss.png", "site_classifier_mlp.pt",
              "site_likelihood.csv"]:
        print(f"    {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
