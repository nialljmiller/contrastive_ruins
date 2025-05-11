import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

# Import project modules
from data_loader import load_shapefiles, load_site_locations, get_raster_paths, sample_random_patches
from data_preparation import create_contrastive_pairs, load_contrastive_pairs
from model import train_siamese_model, load_models
from feature_extraction import extract_features, detect_anomalies
from evaluation import evaluate_results, visualize_results, create_results_report

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Archaeological Ruins Detection')
    parser.add_argument('--data_path', type=str, default='/media/usb',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='output',
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
    
    # Get raster paths
    print("Finding raster files...")
    raster_paths = get_raster_paths(args.data_path, limit=args.raster_limit)
    print(f"Found {len(raster_paths)} raster files")
    
    # Training mode
    if args.mode in ['train', 'full']:
        print("\n--- Training Mode ---")
        
        # Sample random patches
        print(f"Sampling {args.n_samples} random patches from {len(raster_paths)} raster files...")
        patches, patch_locations, _ = sample_random_patches(
            raster_paths, 
            patch_size=args.patch_size, 
            n_samples=args.n_samples, 
            save_dir=patch_dir
        )
        
        # Create contrastive pairs
        print("Creating contrastive pairs...")
        X1, X2, labels = create_contrastive_pairs(
            patches, 
            patch_locations, 
            sites,
            save_dir=patch_dir
        )
        
        # Determine input shape
        if len(X1.shape) == 4:
            input_shape = X1.shape[1:]
        else:
            input_shape = (args.patch_size, args.patch_size, 1)
        
        # Train model
        print(f"Training Siamese network with contrastive loss...")
        encoder, siamese_model, history = train_siamese_model(
            X1, X2, labels,
            input_shape=input_shape,
            epochs=args.epochs,
            save_dir=model_dir
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
        
        # Extract features
        print("Extracting features from test raster...")
        features, locations, _ = extract_features(
            encoder,
            test_raster,
            patch_size=args.patch_size,
            stride=args.patch_size // 2,
            save_dir=results_dir
        )
        
        if len(features) == 0:
            print("Error: No features extracted from test raster!")
            return
        
        # Detect anomalies
        print(f"Detecting potential ruins using {args.detection_method}...")
        ruins_gdf, _ = detect_anomalies(
            features,
            locations,
            method=args.detection_method,
            visualize=True,
            save_dir=results_dir
        )
        
        # Evaluate results