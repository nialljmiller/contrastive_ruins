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


def plot_latent_space(encoder, patches, patch_sources=None, output_dir="results", prefix="latent_space"):
    """Plot 2D and 3D PCA projections of encoder embeddings."""
    device = torch.device("cpu")
    if isinstance(encoder, torch.nn.Module):
        device = next(encoder.parameters()).device
        encoder.eval()

    # Compute embeddings in batches to avoid memory issues
    features = []
    batch_size = 64
    for start in range(0, len(patches), batch_size):
        batch = _prepare_batch(patches[start:start + batch_size])
        batch = batch.to(device)
        with torch.no_grad():
            emb = encoder(batch).cpu().numpy()
        features.append(emb)
    features = np.concatenate(features, axis=0)

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
