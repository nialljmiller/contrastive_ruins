import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns



def plot_latent_space(encoder, patches, patch_sources=None, output_dir="results", prefix="latent_space", precomputed=False):
    """Plot 2D and 3D PCA projections of encoder embeddings."""

    if precomputed:
        features = patches
    else:
        features = compute_embeddings(encoder, patches)

    # 2D PCA
    pca2 = PCA(n_components=2)
    reduced2 = pca2.fit_transform(features)
    plt.figure(figsize=(8, 6))
    if patch_sources is not None:
        unique_sources = sorted(set(patch_sources))
        for src in unique_sources:
            idx = [i for i, s in enumerate(patch_sources) if s == src]
            plt.scatter(reduced2[idx, 0], reduced2[idx, 1], s=5, alpha=0.6, label=src)
        plt.legend(fontsize="small")
    else:
        plt.scatter(reduced2[:, 0], reduced2[:, 1], s=5, alpha=0.6)
    plt.title("Latent Space (2D PCA)")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_2d.png"))
    plt.close()

    # 3D PCA
    pca3 = PCA(n_components=3)
    reduced3 = pca3.fit_transform(features)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if patch_sources is not None:
        for src in unique_sources:
            idx = [i for i, s in enumerate(patch_sources) if s == src]
            ax.scatter(reduced3[idx, 0], reduced3[idx, 1], reduced3[idx, 2], s=5, alpha=0.6, label=src)
        ax.legend(fontsize="small")
    else:
        ax.scatter(reduced3[:, 0], reduced3[:, 1], reduced3[:, 2], s=5, alpha=0.6)
    ax.set_title("Latent Space (3D PCA)")
    plt.savefig(os.path.join(output_dir, f"{prefix}_3d.png"))
    plt.close()

    return features

 
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



def plot_latent_overlay(
    base_features,
    overlay_features,
    base_sources=None,
    overlay_sources=None,
    output_dir="results",
    prefix="latent_overlay",
    max_tsne_points=1000,
    random_state=42,
):
    """Plot PCA, t-SNE and UMAP projections with overlays - IMPROVED READABILITY."""

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
    
    # Use better colors - more distinct and visible
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {src: colors[i % len(colors)] for i, src in enumerate(unique_sources)}

    print("UMAP")
    # ----- 2D UMAP (IMPROVED) -----
    umap2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    base_umap2 = umap2.fit_transform(base_features)
    overlay_umap2 = umap2.transform(overlay_features) if len(overlay_features) > 0 else np.empty((0, 2))

    # Create figure with better size and DPI
    plt.figure(figsize=(12, 9), dpi=150)
    
    # Plot base features with better visibility
    for i, src in enumerate(unique_sources):
        base_idx = [j for j, s in enumerate(base_sources) if s == src]
        if base_idx:
            plt.scatter(base_umap2[base_idx, 0], base_umap2[base_idx, 1], 
                       s=8, alpha=0.7, color=color_map[src], label=src,
                       edgecolors='none')  # Remove edge lines for cleaner look
    
    # Plot overlay features more prominently
    for i, src in enumerate(unique_sources):
        over_idx = [j for j, s in enumerate(overlay_sources) if s == src]
        if over_idx:
            plt.scatter(overlay_umap2[over_idx, 0], overlay_umap2[over_idx, 1], 
                       s=60, marker="X", color=color_map[src], 
                       label=f"{src} (sites)", alpha=0.9,
                       edgecolors='black', linewidth=0.5)
    
    # Improve legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=True, fancybox=True, shadow=True)
    
    plt.title("Latent Space (2D UMAP)", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_umap_2d.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()

    print("3D UMAP")
    # ----- 3D UMAP (IMPROVED) -----
    umap3 = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    base_umap3 = umap3.fit_transform(base_features)
    overlay_umap3 = umap3.transform(overlay_features) if len(overlay_features) > 0 else np.empty((0, 3))

    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot base features
    for i, src in enumerate(unique_sources):
        base_idx = [j for j, s in enumerate(base_sources) if s == src]
        if base_idx:
            ax.scatter(base_umap3[base_idx, 0], base_umap3[base_idx, 1], base_umap3[base_idx, 2],
                      s=8, alpha=0.6, color=color_map[src], label=src)
    
    # Plot overlay features more prominently
    for i, src in enumerate(unique_sources):
        over_idx = [j for j, s in enumerate(overlay_sources) if s == src]
        if over_idx:
            ax.scatter(overlay_umap3[over_idx, 0], overlay_umap3[over_idx, 1], overlay_umap3[over_idx, 2],
                      s=80, marker="X", color=color_map[src], 
                      label=f"{src} (sites)", alpha=0.9,
                      edgecolors='black', linewidth=0.5)
    
    # Improve 3D plot appearance
    ax.set_xlabel("UMAP Dim 1", fontsize=12)
    ax.set_ylabel("UMAP Dim 2", fontsize=12)
    ax.set_zlabel("UMAP Dim 3", fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             fontsize=9, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    ax.set_title("Latent Space (3D UMAP)", fontsize=14, fontweight='bold')
    
    # Set better viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_umap_3d.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()




    # ----- 2D t-SNE (IMPROVED) -----
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

    tsne2 = TSNE(n_components=2, init="pca", random_state=random_state, 
                 perplexity=min(30, (combined.shape[0]-1)/4))
    combined_2d = tsne2.fit_transform(combined)
    base_tsne2 = combined_2d[np.array(base_mask)]
    overlay_tsne2 = combined_2d[~np.array(base_mask)]

    plt.figure(figsize=(12, 9), dpi=150)
    
    for i, src in enumerate(unique_sources):
        base_idx = [j for j, s in enumerate(base_sources_tsne) if s == src]
        if base_idx:
            plt.scatter(base_tsne2[base_idx, 0], base_tsne2[base_idx, 1], 
                       s=8, alpha=0.7, color=color_map[src], label=src,
                       edgecolors='none')
    
    for i, src in enumerate(unique_sources):
        over_idx = [j for j, s in enumerate(overlay_sources_tsne) if s == src]
        if over_idx:
            plt.scatter(overlay_tsne2[over_idx, 0], overlay_tsne2[over_idx, 1], 
                       s=60, marker="X", color=color_map[src], 
                       label=f"{src} (sites)", alpha=0.9,
                       edgecolors='black', linewidth=0.5)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=True, fancybox=True, shadow=True)
    
    plt.title("Latent Space (2D t-SNE)", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_tsne_2d.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()

    return base_umap2, overlay_umap2


def plot_latent_space_density(encoder, patches, patch_sources=None, output_dir="results", prefix="latent_density"):
    """Create density plots for better visualization of overlapping points."""
    
    print("PCA")

    features = compute_embeddings(encoder, patches)
    
    # 2D PCA for density plot
    pca2 = PCA(n_components=2)
    reduced2 = pca2.fit_transform(features)
    
    plt.figure(figsize=(12, 5))
    
    # Regular scatter plot
    plt.subplot(1, 2, 1)
    if patch_sources is not None:
        unique_sources = sorted(set(patch_sources))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, src in enumerate(unique_sources):
            idx = [j for j, s in enumerate(patch_sources) if s == src]
            plt.scatter(reduced2[idx, 0], reduced2[idx, 1], 
                       s=5, alpha=0.6, label=src, color=colors[i % len(colors)])
        plt.legend(fontsize=10)
    else:
        plt.scatter(reduced2[:, 0], reduced2[:, 1], s=5, alpha=0.6)
    plt.title("Scatter Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # Density plot
    plt.subplot(1, 2, 2)
    plt.hexbin(reduced2[:, 0], reduced2[:, 1], gridsize=50, cmap='Blues', mincnt=1)
    plt.colorbar(label='Point Density')
    plt.title("Density Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_density.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()


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


def _prepare_batch(patches):
    batch = patches.astype(np.float32)
    if batch.ndim == 3:  # N,H,W
        batch = np.expand_dims(batch, 1)
    elif batch.ndim == 4 and batch.shape[-1] in {1, 3}:  # N,H,W,C -> N,C,H,W
        batch = np.transpose(batch, (0, 3, 1, 2))
    return torch.from_numpy(batch)