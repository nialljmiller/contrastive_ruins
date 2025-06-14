import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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


def plot_latent_overlay(base_features, overlay_features, output_dir="results", prefix="latent_overlay", labels=("random", "site")):
    """Plot PCA projections with overlays.

    PCA is fit on ``base_features`` and applied to ``overlay_features`` so that
    the orientation of the latent space is determined solely by the random
    samples. Overlay points are then plotted on top using a different marker.
    """

    os.makedirs(output_dir, exist_ok=True)

    # 2D PCA
    pca2 = PCA(n_components=2)
    base_2d = pca2.fit_transform(base_features)
    overlay_2d = pca2.transform(overlay_features) if len(overlay_features) > 0 else np.empty((0, 2))

    plt.figure(figsize=(8, 6))
    plt.scatter(base_2d[:, 0], base_2d[:, 1], s=5, alpha=0.6, color="gray", label=labels[0])
    if len(overlay_2d) > 0:
        plt.scatter(overlay_2d[:, 0], overlay_2d[:, 1], s=20, color="red", marker="x", label=labels[1])
    plt.legend(fontsize="small")
    plt.title("Latent Space (2D PCA)")
    plt.savefig(os.path.join(output_dir, f"{prefix}_2d.png"))
    plt.close()

    # 3D PCA
    pca3 = PCA(n_components=3)
    base_3d = pca3.fit_transform(base_features)
    overlay_3d = pca3.transform(overlay_features) if len(overlay_features) > 0 else np.empty((0, 3))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(base_3d[:, 0], base_3d[:, 1], base_3d[:, 2], s=5, alpha=0.6, color="gray", label=labels[0])
    if len(overlay_3d) > 0:
        ax.scatter(overlay_3d[:, 0], overlay_3d[:, 1], overlay_3d[:, 2], s=20, color="red", marker="x", label=labels[1])
    ax.legend(fontsize="small")
    ax.set_title("Latent Space (3D PCA)")
    plt.savefig(os.path.join(output_dir, f"{prefix}_3d.png"))
    plt.close()

    return base_2d, overlay_2d
