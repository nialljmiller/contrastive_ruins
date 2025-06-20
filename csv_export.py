import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def export_embeddings_to_csv(features, locations, patch_sources=None, output_path="embeddings.csv"):
    """
    Export all embeddings (original features + PCA + t-SNE + UMAP) with coordinates to CSV
    
    Args:
        features: Original feature vectors (N x feature_dim)
        locations: List of shapely geometries for patch locations
        patch_sources: Optional list of source raster names for each patch
        output_path: Path to save CSV file
    """
    
    # Extract spatial coordinates from shapely geometries
    x_coords = []
    y_coords = []
    x_min_coords = []
    y_min_coords = []
    x_max_coords = []
    y_max_coords = []
    
    for geom in locations:
        # Get center coordinates
        x_coords.append(geom.centroid.x)
        y_coords.append(geom.centroid.y)
        
        # Get bounding box coordinates
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        x_min_coords.append(bounds[0])
        y_min_coords.append(bounds[1]) 
        x_max_coords.append(bounds[2])
        y_max_coords.append(bounds[3])
    
    # Compute dimensionality reductions
    print("Computing PCA...")
    pca_2d = PCA(n_components=2, random_state=69)
    pca_3d = PCA(n_components=3, random_state=69)
    pca_2d_coords = pca_2d.fit_transform(features)
    pca_3d_coords = pca_3d.fit_transform(features)
    
    print("Computing t-SNE...")
    tsne_2d = TSNE(n_components=2, random_state=69, init='pca')
    tsne_3d = TSNE(n_components=3, random_state=69, init='pca')
    tsne_2d_coords = tsne_2d.fit_transform(features)
    tsne_3d_coords = tsne_3d.fit_transform(features)
    
    print("Computing UMAP...")
    umap_2d = umap.UMAP(n_components=2, random_state=69)
    umap_3d = umap.UMAP(n_components=3, random_state=69)
    umap_2d_coords = umap_2d.fit_transform(features)
    umap_3d_coords = umap_3d.fit_transform(features)
    
    # Create DataFrame
    data_dict = {
        # Spatial coordinates
        'x_center': x_coords,
        'y_center': y_coords,
        'x_min': x_min_coords,
        'y_min': y_min_coords,
        'x_max': x_max_coords,
        'y_max': y_max_coords,
        
        # PCA coordinates
        'pca_1': pca_2d_coords[:, 0],
        'pca_2': pca_2d_coords[:, 1],
        'pca_3': pca_3d_coords[:, 2],
        
        # t-SNE coordinates  
        'tsne_1': tsne_2d_coords[:, 0],
        'tsne_2': tsne_2d_coords[:, 1],
        'tsne_3': tsne_3d_coords[:, 2],
        
        # UMAP coordinates
        'umap_1': umap_2d_coords[:, 0],
        'umap_2': umap_2d_coords[:, 1], 
        'umap_3': umap_3d_coords[:, 2],
    }
    
    # Add patch sources if provided
    if patch_sources is not None:
        data_dict['source_raster'] = patch_sources
    
    # Add original feature dimensions
    for i in range(features.shape[1]):
        data_dict[f'feature_{i:03d}'] = features[:, i]
    
    # Create DataFrame and save
    df = pd.DataFrame(data_dict)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df)} embeddings to {output_path}")
    print(f"Columns: {list(df.columns)}")
    
    return df