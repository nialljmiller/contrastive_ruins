import numpy as np
import rasterio
import os
from shapely.geometry import box
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import geopandas as gpd

def normalize_patch(patch):
    """Normalize patch to 0-1 range"""
    patch_min = np.min(patch)
    patch_max = np.max(patch)
    
    if patch_max > patch_min:
        normalized = (patch - patch_min) / (patch_max - patch_min)
    else:
        normalized = np.zeros_like(patch)
    
    return normalized


def extract_features(encoder, raster_path, patch_size=256, stride=128, batch_size=16, save_dir=None):
    """
    Extract features from a raster using sliding window approach - MEMORY OPTIMIZED
    """
    features = []
    locations = []

    # Determine device from encoder and set eval mode if applicable
    device = torch.device("cpu")
    if isinstance(encoder, torch.nn.Module):
        device = next(encoder.parameters()).device
        encoder.to(device)
        encoder.eval()
    
    try:
        with rasterio.open(raster_path) as src:
            # Get raster dimensions
            height, width = src.height, src.width
            nodata_value = src.nodata
            
            # Prepare batches for more efficient processing
            current_batch = []
            current_locations = []
            
            # Slide window over raster
            for row in tqdm(range(0, height - patch_size, stride), 
                           desc=f"Extracting features from {os.path.basename(raster_path)}"):
                for col in range(0, width - patch_size, stride):
                    # Read patch
                    window = ((row, row + patch_size), (col, col + patch_size))
                    patch = src.read(window=window)
                    
                    # Skip patches with no data or all zeros
                    if nodata_value is not None and np.any(patch == nodata_value):
                        continue
                    if np.std(patch) < 0.01 or np.all(patch == 0):
                        continue
                    
                    # Handle different band configurations
                    if patch.shape[0] == 1:  # Single band data
                        patch = patch[0]  # Remove band dimension
                    elif patch.shape[0] == 3:  # Three band data
                        patch = np.transpose(patch, (1, 2, 0))  # Change to HWC format
                    
                    # Normalize patch
                    patch = normalize_patch(patch)
                    
                    # Get coordinates
                    x_min, y_max = src.transform * (col, row)
                    x_max, y_min = src.transform * (col + patch_size, row + patch_size)
                    geometry = box(x_min, y_min, x_max, y_max)
                    
                    # Add to current batch
                    current_batch.append(patch)
                    current_locations.append(geometry)
                    # REMOVED: all_patches.append(patch)  # This was eating memory!
                    
                    # Process batch when it reaches the specified size
                    if len(current_batch) >= batch_size:
                        batch_array = np.array(current_batch)
                        if len(batch_array.shape) == 3:
                            batch_array = np.expand_dims(batch_array, axis=-1)
                        batch_tensor = torch.from_numpy(batch_array).float()
                        if batch_tensor.ndim == 4:
                            batch_tensor = batch_tensor.permute(0, 3, 1, 2)
                        batch_tensor = batch_tensor.to(device)
                        with torch.no_grad():
                            batch_features = encoder(batch_tensor).cpu().numpy()
                        features.extend(batch_features)
                        locations.extend(current_locations)
                        
                        # Reset batch AND clear memory
                        current_batch = []
                        current_locations = []
                        del batch_array, batch_features  # Explicit cleanup
            
            # Process remaining patches
            if current_batch:
                batch_array = np.array(current_batch)
                if len(batch_array.shape) == 3:
                    batch_array = np.expand_dims(batch_array, axis=-1)
                batch_tensor = torch.from_numpy(batch_array).float()
                if batch_tensor.ndim == 4:
                    batch_tensor = batch_tensor.permute(0, 3, 1, 2)
                batch_tensor = batch_tensor.to(device)
                with torch.no_grad():
                    batch_features = encoder(batch_tensor).cpu().numpy()
                features.extend(batch_features)
                locations.extend(current_locations)
                
                # Clean up
                del batch_array, batch_features
    
    except Exception as e:
        print(f"Error extracting features from {raster_path}: {e}")
        return [], [], []
    
    features = np.array(features)
    
    # Save features if directory is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(raster_path))[0]
        np.save(os.path.join(save_dir, f"{basename}_features.npy"), features)
    
    print(f"Extracted {len(features)} feature vectors from {os.path.basename(raster_path)}")
    return features, locations, []  # Return empty list instead of all_patches

def detect_anomalies(features, locations, method='iforest', contamination=0.05, 
                     n_clusters=5, visualize=True, save_dir=None):
    """
    Detect anomalies (potential ruins) in extracted features
    
    Args:
        features: Extracted feature vectors
        locations: Geometries of patch locations
        method: Detection method ('iforest' or 'kmeans')
        contamination: Expected proportion of anomalies (for IsolationForest)
        n_clusters: Number of clusters (for KMeans)
        visualize: Whether to visualize results
        save_dir: Directory to save results
        
    Returns:
        ruins_gdf: GeoDataFrame with detected ruins
        scores: Anomaly scores or cluster assignments
    """
    # Reduce dimensionality for visualization
    if features.shape[1] > 2 and visualize:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
    else:
        features_2d = features
    
    # Detect anomalies
    if method == 'iforest':
        # Isolation Forest for anomaly detection
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        predictions = model.fit_predict(features)
        scores = model.decision_function(features)
        
        # Anomalies have prediction -1
        anomaly_indices = np.where(predictions == -1)[0]
        potential_ruins = [locations[i] for i in anomaly_indices]
        
        # Create GeoDataFrame with anomaly scores
        ruins_data = {
            'anomaly_score': scores[anomaly_indices],
            'geometry': potential_ruins
        }
        
    elif method == 'kmeans':
        # K-means clustering
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = model.fit_predict(features)
        
        # Calculate distances to cluster centers
        distances = np.min(
            [np.linalg.norm(features - center, axis=1) for center in model.cluster_centers_],
            axis=0
        )
        
        # Find potential ruins (points far from their cluster centers)
        threshold = np.percentile(distances, 100 - contamination * 100)
        anomaly_indices = np.where(distances > threshold)[0]
        potential_ruins = [locations[i] for i in anomaly_indices]
        
        # Create GeoDataFrame with cluster and distance information
        ruins_data = {
            'cluster': clusters[anomaly_indices],
            'distance': distances[anomaly_indices],
            'geometry': potential_ruins
        }
        scores = clusters
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iforest' or 'kmeans'.")
    
    # Create GeoDataFrame
    ruins_gdf = gpd.GeoDataFrame(ruins_data)
    
    # Visualize results if requested
    if visualize:
        plt.figure(figsize=(10, 8))
        
        if method == 'iforest':
            # Plot all points
            plt.scatter(
                features_2d[:, 0], features_2d[:, 1], 
                c=predictions == -1, cmap='coolwarm',
                alpha=0.5, s=10
            )
            plt.colorbar(label='Anomaly')
            plt.title(f'Isolation Forest Anomaly Detection\n{len(anomaly_indices)} potential ruins')
            
        elif method == 'kmeans':
            # Plot clusters
            plt.scatter(
                features_2d[:, 0], features_2d[:, 1], 
                c=clusters, cmap='viridis',
                alpha=0.5, s=10
            )
            plt.colorbar(label='Cluster')
            
            # Highlight anomalies
            plt.scatter(
                features_2d[anomaly_indices, 0], features_2d[anomaly_indices, 1],
                c='red', s=20, alpha=0.8, marker='x'
            )
            plt.title(f'K-means Clustering Anomaly Detection\n{len(anomaly_indices)} potential ruins')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{method}_detection.png"))
        else:
            plt.show()
        plt.close()
    
    print(f"Detected {len(ruins_gdf)} potential ruins using {method}")
    return ruins_gdf, scores

if __name__ == "__main__":
    print("Feature extraction module - use in main.py")
