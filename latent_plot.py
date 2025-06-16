import os
import argparse
import numpy as np

from data_loader import (
    load_shapefiles,
    load_site_locations,
    get_raster_paths,
    sample_random_patches,
    sample_site_patches,
)
from model import load_models
from visualization import compute_embeddings, plot_latent_space


def parse_args():
    parser = argparse.ArgumentParser(description="Plot encoder latent space with known sites")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/project/joycelab-niall/ruin_repo",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="latent_results",
        help="Directory to save latent space plots",
    )
    parser.add_argument("--raster_limit", type=int, default=3, help="Number of rasters to sample")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of patches")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of random patches")
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Directory containing trained models"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    shapefiles = load_shapefiles(args.data_path)
    known_sites = load_site_locations(args.data_path)
    if not hasattr(known_sites, "geometry"):
        known_sites = shapefiles["sites"]

    raster_paths = get_raster_paths(args.data_path, limit=args.raster_limit)

    # Sample patches
    patches, _, _ = sample_random_patches(
        raster_paths, patch_size=args.patch_size, n_samples=args.n_samples
    )
    site_patches, _, _ = sample_site_patches(raster_paths, known_sites, patch_size=args.patch_size)

    # Load encoder
    encoder, _ = load_models(save_dir=args.model_dir)

    # Compute embeddings
    patch_emb = compute_embeddings(encoder, patches)
    site_emb = compute_embeddings(encoder, site_patches) if len(site_patches) > 0 else np.empty((0, patch_emb.shape[1]))

    all_emb = np.concatenate([patch_emb, site_emb]) if len(site_emb) > 0 else patch_emb
    labels = ["raster"] * len(patch_emb) + ["known_site"] * len(site_emb)

    # Plot latent space
    plot_latent_space(
        encoder,
        all_emb,
        patch_sources=labels if labels else None,
        output_dir=args.output_dir,
        prefix="latent_with_sites",
        precomputed=True,
    )


if __name__ == "__main__":
    main()
