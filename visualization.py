import matplotlib.pyplot as plt
import numpy as np


def plot_latent_overlay(base_pca, overlay_dict, save_path=None):
    """Plot 2D latent space scatter with overlays.

    Parameters
    ----------
    base_pca : np.ndarray
        Nx2 array of base latent coordinates.
    overlay_dict : dict[str, np.ndarray]
        Mapping from label to Mx2 array of overlay coordinates.
    save_path : str, optional
        If provided, save the figure instead of showing it.
    """
    if base_pca.ndim != 2 or base_pca.shape[1] != 2:
        raise ValueError("base_pca must be a (N,2) array")

    plt.figure(figsize=(8, 6))
    plt.scatter(base_pca[:, 0], base_pca[:, 1], s=5, alpha=0.4, color="gray", label="base")

    colors = plt.cm.tab10.colors
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(overlay_dict)}

    for i, (label, coords) in enumerate(overlay_dict.items()):
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("overlay coordinates must be (M,2) arrays")
        plt.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.6, color=color_map[label], label=label)

    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    base = np.random.randn(100, 2)
    overlay = {"set1": np.random.randn(30, 2) + 2,
               "set2": np.random.randn(30, 2) - 2}
    plot_latent_overlay(base, overlay)

