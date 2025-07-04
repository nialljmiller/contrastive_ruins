import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import project modules
from csv_export import export_embeddings_to_csv  # Add this line
from data_loader import load_shapefiles, load_site_locations, get_raster_paths, sample_random_patches, sample_site_patches
from data_preparation import create_contrastive_pairs, load_contrastive_pairs
from model import train_siamese_model, load_models
from feature_extraction import extract_features, detect_anomalies
from evaluation import evaluate_results, visualize_results, create_results_report
from visualization import (
    plot_latent_space,
    plot_training_loss,
    compute_embeddings,
    plot_latent_overlay,
)


def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Archaeological Ruins Detection')
    parser.add_argument('--data_path', type=str, default='/project/galacticbulge/ruin_repo',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='/project/galacticbulge/ruin_repo/output',
                        help='Directory to save results')
    parser.add_argument('--mode', type=str, choices=['train', 'detect', 'full'], default='full',
                        help='Mode: train model, detect ruins, or run full pipeline')
    parser.add_argument('--raster_limit', type=int, default=3,
                        help='Maximum number of raster files to process')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of image patches')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of patches to sample for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--latent_interval', type=int, default=100,
                        help='Epoch interval for UMAP/t-SNE/PCA plots')
    parser.add_argument('--latent_sample_size', type=int, default=1000,
                        help='Number of patches to sample for latent plots')
    parser.add_argument('--latent_quick_interval', type=int, default=10,
                        help='Epoch interval for lightweight latent plots')
    parser.add_argument('--latent_quick_size', type=int, default=200,
                        help='Number of patches for quick latent plots')
    parser.add_argument('--detection_method', type=str, choices=['iforest', 'kmeans'], default='iforest',
                        help='Method for anomaly detection')
    parser.add_argument('--test_tile', type=str, default=None,
                        help='Specific raster file to use for detection')
    
    return parser.parse_args()

def main():
    """Main function to run the pipeline"""
    # Parse arguments
    args = setup_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up directories
    patch_dir = os.path.join(args.output_dir, 'patches')
    model_dir = os.path.join(args.output_dir, 'models')
    results_dir = os.path.join(args.output_dir, 'results')
    
    print(f"Archaeological Ruins Detection Pipeline")
    print(f"======================================")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Detection method: {args.detection_method}")
    print()
    
    # Load shapefiles
    print("Loading shapefiles...")
    shapefiles = load_shapefiles(args.data_path)
    sites = shapefiles['sites']
    study_area = shapefiles['study_area']
    lidar_areas = shapefiles['lidar_areas']
    
    # Load known site locations
    print("Loading known site locations...")
    known_sites = load_site_locations(args.data_path)
    if not hasattr(known_sites, "geometry"):
        print("Site CSV lacks geometry; using shapefile sites instead")
        known_sites = sites
    
    # Get raster paths
    print("Finding raster files...")
    raster_paths = get_raster_paths(args.data_path, limit=args.raster_limit)
    print(f"Found {len(raster_paths)} raster files")
    
    # Training mode
    if args.mode in ['train', 'full']:
        print("\n--- Training Mode ---")
        
        # Sample random patches
        print(f"Sampling {args.n_samples} random patches from {len(raster_paths)} raster files...")
        patches, patch_locations, patch_sources = sample_random_patches(
            raster_paths,
            patch_size=args.patch_size,
            n_samples=args.n_samples,
            save_dir=patch_dir
        )

        # Extract patches around known archaeological sites for context
        site_patches, _, _ = sample_site_patches(
            raster_paths,
            known_sites,
            patch_size=args.patch_size,
            save_dir=patch_dir
        )

        all_patches = np.concatenate([patches, site_patches]) if len(site_patches) > 0 else patches
        all_sources = patch_sources + ["known_site"] * len(site_patches)
        
        # Create contrastive pairs for SELF-SUPERVISED learning
        print("Creating self-supervised contrastive pairs...")
        print("Note: Not using known site locations - this is purely self-supervised")
        X1, X2, labels = create_contrastive_pairs(
            patches, 
            patch_locations, 
            sites_gdf=None,  # Don't use sites for self-supervised learning
            save_dir=patch_dir
        )

        # Train model
        print(f"Training Siamese network with contrastive loss...")
        encoder, siamese_model, history = train_siamese_model(
            X1, X2, labels,
            epochs=args.epochs,
            save_dir=model_dir,
            checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
            latent_interval=args.latent_interval,
            latent_sample_size=args.latent_sample_size,
            latent_quick_interval=args.latent_quick_interval,
            latent_quick_size=args.latent_quick_size,
        )

        plot_training_loss(history, os.path.join(results_dir, "training_loss.png"))

        # Visualize latent space without and with patch source labels
        plot_latent_space(encoder, patches, output_dir=results_dir)
        plot_latent_space(
            encoder,
            patches,
            patch_sources=patch_sources,
            output_dir=results_dir,
            prefix="latent_space_features",
        )
        if len(site_patches) > 0:
            plot_latent_space(
                encoder,
                all_patches,
                patch_sources=all_sources,
                output_dir=results_dir,
                prefix="latent_space_with_sites",
            )
    else:
        # Load pre-trained models
        print("Loading pre-trained models...")
        encoder, siamese_model = load_models(save_dir=model_dir)
    
    # Detection mode
    if args.mode in ['detect', 'full']:
        print("\n--- Detection Mode ---")
        
        # Select test raster
        if args.test_tile:
            test_raster = os.path.join(args.data_path, 'rasters', args.test_tile)
            if not os.path.exists(test_raster):
                print(f"Test raster {test_raster} not found!")
                test_raster = raster_paths[-1]  # Use last raster as fallback
        else:
            # Use a different raster for testing
            test_raster = raster_paths[-1]
        
        print(f"Using {os.path.basename(test_raster)} for detection...")

        # Extract features from all rasters once and retain those for the test
        # raster so we don't compute them twice.
        base_features = []
        base_sources = []
        features = None
        locations = None

        for raster_path in raster_paths:
            print(f"Extracting features from {os.path.basename(raster_path)}...")
            feats, locs, _ = extract_features(
                encoder,
                raster_path,
                patch_size=args.patch_size,
                stride=args.patch_size // 2,
                save_dir=results_dir,
            )
            base_features.extend(feats)
            base_sources.extend([os.path.basename(raster_path)] * len(feats))

            if os.path.samefile(raster_path, test_raster):
                features = np.array(feats)
                locations = locs

        base_features = np.array(base_features)

        if features is None:
            # If the loop didn't match the test raster path, extract now
            features, locations, _ = extract_features(
                encoder,
                test_raster,
                patch_size=args.patch_size,
                stride=args.patch_size // 2,
                save_dir=results_dir,
            )

        if len(features) == 0:
            print("Error: No features extracted from test raster!")
            return

        # Compute embeddings for known sites to plot alongside detection features
        site_patches_det, _, _ = sample_site_patches(
            [test_raster],
            known_sites,
            patch_size=args.patch_size,
        )

        site_features_det = (
            compute_embeddings(encoder, site_patches_det)
            if len(site_patches_det) > 0
            else np.empty((0, features.shape[1]))
        )

        # Visualize the feature distribution with the known site embeddings
        # overlaid. ``base_features`` and ``base_sources`` come from all
        # rasters so they have matching lengths.
        plot_latent_overlay(
            base_features,
            site_features_det,
            base_sources=base_sources,
            overlay_sources=["known_site"] * len(site_features_det),
            output_dir=results_dir,
            prefix="detection_latent_space",
        )
        
        # Detect anomalies
        print(f"Detecting potential ruins using {args.detection_method}...")
        ruins_gdf, _ = detect_anomalies(
            features,
            locations,
            method=args.detection_method,
            visualize=True,
            save_dir=results_dir
        )


        print("Exporting embeddings to CSV...")
        csv_path = os.path.join(results_dir, "embeddings_with_coordinates.csv")
        export_embeddings_to_csv(
            features=features,
            locations=locations, 
            patch_sources=[os.path.basename(test_raster)] * len(features),
            output_path=csv_path
        )

        
if __name__ == "__main__":
    main()
