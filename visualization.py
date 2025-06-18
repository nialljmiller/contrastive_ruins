import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401



def plot_training_loss(history, output_path):
    """Save a plot of training loss over epochs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def _prepare_batch(patches):
    batch = patches.astype(np.float32)
    if batch.ndim == 3:  # N,H,W
        batch = np.expand_dims(batch, 1)
    elif batch.ndim == 4 and batch.shape[-1] in {1, 3}:  # N,H,W,C -> N,C,H,W
        batch = np.transpose(batch, (0, 3, 1, 2))
    return torch.from_numpy(batch)


def compute_embeddings(encoder, patches, batch_size=64):
    """Return encoder embeddings for an array of patches."""
    device = torch.device("cpu")
    if isinstance(encoder, torch.nn.Module):
        device = next(encoder.parameters()).device
        encoder.eval()

    feats = []
    for start in range(0, len(patches), batch_size):
        batch = _prepare_batch(patches[start:start + batch_size])
        batch = batch.to(device)
        with torch.no_grad():
            emb = encoder(batch).cpu().numpy()
        feats.append(emb)
    if feats:
        return np.concatenate(feats, axis=0)
    return np.empty((0, 1))




def plot_latent_overlay(
    base_features,
    overlay_features,
    base_sources=None,
    overlay_sources=None,
    output_dir="results",
    prefix="latent_overlay",
    max_tsne_points=10000,
    random_state=42,
):
    """Plot PCA, t-SNE and UMAP projections with overlays.

    ``base_sources`` and ``overlay_sources`` allow coloring points by their
    originating raster, similar to :func:`plot_latent_space`. PCA and UMAP are
    fit only on ``base_features`` so the random patches determine the orienta-
    tion. t-SNE does not support transforming new samples, so both feature sets
    are concatenated before fitting.
    
    Sources are plotted in order of dataset size (largest to smallest) so that
    larger datasets appear on the bottom layer.
    """

    os.makedirs(output_dir, exist_ok=True)

    base_sources = base_sources or ["base"] * len(base_features)
    if overlay_features is None:
        overlay_features = np.empty((0, base_features.shape[1]))
    overlay_sources = overlay_sources or ["overlay"] * len(overlay_features)
    all_sources = base_sources + overlay_sources
    
    # Count points for each source to sort by size
    source_counts = {}
    for src in set(all_sources):
        base_count = sum(1 for s in base_sources if s == src)
        overlay_count = sum(1 for s in overlay_sources if s == src)
        source_counts[src] = base_count + overlay_count
    
    # Sort sources by total count (largest to smallest)
    unique_sources = sorted(source_counts.keys(), key=lambda x: source_counts[x], reverse=True)
    
    cmap = plt.get_cmap("tab10", len(unique_sources))
    color_map = {src: cmap(i) for i, src in enumerate(unique_sources)}

    # ----- PCA -----
    pca2 = PCA(n_components=2)
    base_pca2 = pca2.fit_transform(base_features)
    overlay_pca2 = pca2.transform(overlay_features) if len(overlay_features) > 0 else np.empty((0, 2))

    plt.figure(figsize=(8, 6))
    for src in unique_sources:  # Now sorted by size
        base_idx = [i for i, s in enumerate(base_sources) if s == src]
        over_idx = [i for i, s in enumerate(overlay_sources) if s == src]
        if base_idx:
            plt.scatter(base_pca2[base_idx, 0], base_pca2[base_idx, 1], s=5, alpha=0.6, color=color_map[src], label=src)
        if over_idx:
            plt.scatter(overlay_pca2[over_idx, 0], overlay_pca2[over_idx, 1], s=20, marker="x", color=color_map[src], label=f"{src}_overlay")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize="small")
    plt.title("Latent Space (2D PCA)")
    plt.savefig(os.path.join(output_dir, f"{prefix}_pca_2d.png"))
    plt.close()

    pca3 = PCA(n_components=3)
    base_pca3 = pca3.fit_transform(base_features)
    overlay_pca3 = pca3.transform(overlay_features) if len(overlay_features) > 0 else np.empty((0, 3))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for src in unique_sources:  # Now sorted by size
        base_idx = [i for i, s in enumerate(base_sources) if s == src]
        over_idx = [i for i, s in enumerate(overlay_sources) if s == src]
        if base_idx:
            ax.scatter(
                base_pca3[base_idx, 0],
                base_pca3[base_idx, 1],
                base_pca3[base_idx, 2],
                s=5,
                alpha=0.6,
                color=color_map[src],
                label=src,
            )
        if over_idx:
            ax.scatter(
                overlay_pca3[over_idx, 0],
                overlay_pca3[over_idx, 1],
                overlay_pca3[over_idx, 2],
                s=20,
                marker="x",
                color=color_map[src],
                label=f"{src}_overlay",
            )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize="small")
    ax.set_title("Latent Space (3D PCA)")
    plt.savefig(os.path.join(output_dir, f"{prefix}_pca_3d.png"))
    plt.close()

    # ----- t-SNE -----
    combined = base_features if len(overlay_features) == 0 else np.vstack([base_features, overlay_features])

    if combined.shape[0] > max_tsne_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(combined.shape[0], max_tsne_points, replace=False)
        combined = combined[idx]
        base_mask = [i < len(base_features) for i in idx]
        base_sources_tsne = [base_sources[i] for i in idx if i < len(base_features)]
        overlay_sources_tsne = [overlay_sources[i - len(base_features)] for i in idx if i >= len(base_features)]
    else:
        base_mask = [i < len(base_features) for i in range(combined.shape[0])]
        base_sources_tsne = base_sources
        overlay_sources_tsne = overlay_sources

    tsne2 = TSNE(n_components=2, init="pca", random_state=random_state)
    combined_2d = tsne2.fit_transform(combined)
    base_tsne2 = combined_2d[np.array(base_mask)]
    overlay_tsne2 = combined_2d[~np.array(base_mask)]

    plt.figure(figsize=(8, 6))
    for src in unique_sources:  # Now sorted by size
        base_idx = [i for i, s in enumerate(base_sources_tsne) if s == src]
        over_idx = [i for i, s in enumerate(overlay_sources_tsne) if s == src]
        if base_idx:
            plt.scatter(base_tsne2[base_idx, 0], base_tsne2[base_idx, 1], s=5, alpha=0.6, color=color_map[src], label=src)
        if over_idx:
            plt.scatter(overlay_tsne2[over_idx, 0], overlay_tsne2[over_idx, 1], s=20, marker="x", color=color_map[src], label=f"{src}_overlay")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize="small")
    plt.title("Latent Space (2D t-SNE)")
    plt.savefig(os.path.join(output_dir, f"{prefix}_tsne_2d.png"))
    plt.close()

    tsne3 = TSNE(n_components=3, init="pca", random_state=random_state)
    combined_3d = tsne3.fit_transform(combined)
    base_tsne3 = combined_3d[np.array(base_mask)]
    overlay_tsne3 = combined_3d[~np.array(base_mask)]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for src in unique_sources:  # Now sorted by size
        base_idx = [i for i, s in enumerate(base_sources_tsne) if s == src]
        over_idx = [i for i, s in enumerate(overlay_sources_tsne) if s == src]
        if base_idx:
            ax.scatter(
                base_tsne3[base_idx, 0],
                base_tsne3[base_idx, 1],
                base_tsne3[base_idx, 2],
                s=5,
                alpha=0.6,
                color=color_map[src],
                label=src,
            )
        if over_idx:
            ax.scatter(
                overlay_tsne3[over_idx, 0],
                overlay_tsne3[over_idx, 1],
                overlay_tsne3[over_idx, 2],
                s=20,
                marker="x",
                color=color_map[src],
                label=f"{src}_overlay",
            )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize="small")
    ax.set_title("Latent Space (3D t-SNE)")
    plt.savefig(os.path.join(output_dir, f"{prefix}_tsne_3d.png"))
    plt.close()

    # ----- UMAP -----
    umap2 = umap.UMAP(n_components=2, random_state=42)
    base_umap2 = umap2.fit_transform(base_features)
    overlay_umap2 = umap2.transform(overlay_features) if len(overlay_features) > 0 else np.empty((0, 2))

    plt.figure(figsize=(8, 6))
    for src in unique_sources:  # Now sorted by size
        base_idx = [i for i, s in enumerate(base_sources) if s == src]
        over_idx = [i for i, s in enumerate(overlay_sources) if s == src]
        if base_idx:
            plt.scatter(base_umap2[base_idx, 0], base_umap2[base_idx, 1], s=5, alpha=0.6, color=color_map[src], label=src)
        if over_idx:
            plt.scatter(overlay_umap2[over_idx, 0], overlay_umap2[over_idx, 1], s=20, marker="x", color=color_map[src], label=f"{src}_overlay")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize="small")
    plt.title("Latent Space (2D UMAP)")
    plt.savefig(os.path.join(output_dir, f"{prefix}_umap_2d.png"))
    plt.close()

    umap3 = umap.UMAP(n_components=3, random_state=42)
    base_umap3 = umap3.fit_transform(base_features)
    overlay_umap3 = umap3.transform(overlay_features) if len(overlay_features) > 0 else np.empty((0, 3))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for src in unique_sources:  # Now sorted by size
        base_idx = [i for i, s in enumerate(base_sources) if s == src]
        over_idx = [i for i, s in enumerate(overlay_sources) if s == src]
        if base_idx:
            ax.scatter(
                base_umap3[base_idx, 0],
                base_umap3[base_idx, 1],
                base_umap3[base_idx, 2],
                s=5,
                alpha=0.6,
                color=color_map[src],
                label=src,
            )
        if over_idx:
            ax.scatter(
                overlay_umap3[over_idx, 0],
                overlay_umap3[over_idx, 1],
                overlay_umap3[over_idx, 2],
                s=20,
                marker="x",
                color=color_map[src],
                label=f"{src}_overlay",
            )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize="small")
    ax.set_title("Latent Space (3D UMAP)")
    plt.savefig(os.path.join(output_dir, f"{prefix}_umap_3d.png"))
    plt.close()

    return base_pca2, overlay_pca2
