import rasterio
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import random

def load_shapefiles(base_path):
    """Load all necessary shapefiles"""
    shapefile_dir = os.path.join(base_path, 'shapefiles')
    
    sites = gpd.read_file(os.path.join(shapefile_dir, 'sites_sensitive.shp'))
    study_area = gpd.read_file(os.path.join(shapefile_dir, 'study_area_grid.shp'))
    exclusion_areas = gpd.read_file(os.path.join(shapefile_dir, 'Reservation_exclusion.shp'))
    lidar_areas = gpd.read_file(os.path.join(shapefile_dir, 'areas_with_lidar.shp'))
    
    return {
        'sites': sites,
        'study_area': study_area,
        'exclusion_areas': exclusion_areas,
        'lidar_areas': lidar_areas
    }

def load_site_locations(base_path):
    """Load known site locations CSV"""
    site_locations_dir = os.path.join(base_path, 'site_locations')
    known_sites = pd.read_csv(os.path.join(site_locations_dir, 'site_locations_mvnp.csv'))
    
    # Convert to GeoDataFrame if coordinates are available
    if 'longitude' in known_sites.columns and 'latitude' in known_sites.columns:
        known_sites_gdf = gpd.GeoDataFrame(
            known_sites, 
            geometry=gpd.points_from_xy(known_sites.longitude, known_sites.latitude)
        )
        return known_sites_gdf
    else:
        print("Warning: Site locations CSV does not have latitude/longitude columns")
        return known_sites

def get_raster_paths(base_path, limit=None):
    """Get paths for all raster TIFs, with optional limit"""
    raster_dir = os.path.join(base_path, 'rasters')
    raster_files = [f for f in os.listdir(raster_dir) if f.endswith('.tif')]
    
    if limit is not None:
        raster_files = raster_files[:limit]
    
    raster_paths = [os.path.join(raster_dir, f) for f in raster_files]
    return raster_paths

def sample_random_patches(raster_paths, patch_size=256, n_samples=10000, save_dir=None):
    """
    Sample random patches from raster files
    
    Args:
        raster_paths: List of paths to raster files
        patch_size: Size of patches to extract (square)
        n_samples: Total number of patches to extract
        save_dir: Directory to save extracted patches (optional)
        
    Returns:
        patches: Array of extracted patches
        patch_locations: List of bounding boxes as (x_min, y_min, x_max, y_max)
    """
    patches = []
    patch_locations = []
    patch_sources = []  # Track which file each patch came from
    
    patches_per_raster = n_samples // len(raster_paths)
    
    for i, raster_path in enumerate(tqdm(raster_paths, desc="Sampling patches")):
        try:
            with rasterio.open(raster_path) as src:
                height, width = src.height, src.width
                
                for j in range(patches_per_raster):
                    # Generate random location within raster bounds
                    row = random.randint(0, height - patch_size - 1)
                    col = random.randint(0, width - patch_size - 1)
                    
                    # Read patch
                    window = ((row, row + patch_size), (col, col + patch_size))
                    patch = src.read(window=window)
                    
                    # Skip if patch contains no data values
                    if np.any(patch == src.nodata) or np.all(patch == 0):
                        continue
                    
                    # Normalize patch (assuming single band)
                    if patch.shape[0] == 1:
                        patch = patch[0]  # Remove band dimension for single-band data
                    
                    # Get coordinates
                    x_min, y_max = src.transform * (col, row)
                    x_max, y_min = src.transform * (col + patch_size, row + patch_size)
                    
                    patches.append(patch)
                    patch_locations.append((x_min, y_min, x_max, y_max))
                    patch_sources.append(os.path.basename(raster_path))
                    
                    # Save patch if directory is provided
                    if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        patch_filename = f"patch_{i}_{j}.npy"
                        np.save(os.path.join(save_dir, patch_filename), patch)
        
        except Exception as e:
            print(f"Error processing {raster_path}: {e}")
            continue
    
    print(f"Successfully extracted {len(patches)} patches")
    return np.array(patches), patch_locations, patch_sources

def sample_site_patches(raster_paths, sites_gdf, patch_size=256, save_dir=None):
    """Extract patches centered on known site locations.

    Returns:
        patches: Array of extracted patches
        patch_locations: List of bounding boxes as (x_min, y_min, x_max, y_max)
        patch_sources: Which raster each patch came from
    """

    if {"longitude", "latitude"}.issubset(sites_gdf.columns):
        coords = list(zip(sites_gdf.longitude, sites_gdf.latitude))
    elif "geometry" in sites_gdf.columns:
        coords = [(geom.x, geom.y) for geom in sites_gdf.geometry]
    else:
        print(
            "Known site locations lack geometry or lat/long columns; skipping site patch extraction."
        )
        return np.empty((0, patch_size, patch_size)), [], []

    patches = []
    patch_locations = []
    patch_sources = []

    for idx, (lon, lat) in enumerate(coords):
        for raster_path in raster_paths:
            try:
                with rasterio.open(raster_path) as src:
                    left, bottom, right, top = src.bounds
                    if not (left <= lon <= right and bottom <= lat <= top):
                        continue

                    row, col = src.index(lon, lat)

                    row_start = max(0, row - patch_size // 2)
                    col_start = max(0, col - patch_size // 2)
                    row_start = min(row_start, src.height - patch_size)
                    col_start = min(col_start, src.width - patch_size)

                    window = ((row_start, row_start + patch_size), (col_start, col_start + patch_size))
                    patch = src.read(window=window)

                    if np.any(patch == src.nodata) or np.all(patch == 0):
                        continue

                    if patch.shape[0] == 1:
                        patch = patch[0]

                    x_min, y_max = src.transform * (col_start, row_start)
                    x_max, y_min = src.transform * (col_start + patch_size, row_start + patch_size)

                    patches.append(patch)
                    patch_locations.append((x_min, y_min, x_max, y_max))

                    patch_sources.append(os.path.basename(raster_path))

                    if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        patch_filename = f"site_patch_{idx}.npy"
                        np.save(os.path.join(save_dir, patch_filename), patch)

                    break
            except Exception as e:
                print(f"Error extracting site patch from {raster_path}: {e}")
                continue

    print(f"Extracted {len(patches)} site patches")
    return np.array(patches), patch_locations, patch_sources

if __name__ == "__main__":
    # Test the module
    base_path = '/project/galacticbulge/ruin_repo/'
    shapefiles = load_shapefiles(base_path)
    known_sites = load_site_locations(base_path)
    raster_paths = get_raster_paths(base_path, limit=2)  # Test with first 2 rasters
    
    print(f"Loaded {len(shapefiles['sites'])} sites")
    print(f"Loaded {len(known_sites)} known site locations")
    print(f"Found {len(raster_paths)} raster files")
    
    # Test patch extraction with a small sample
    patches, locations, sources = sample_random_patches(
        raster_paths, patch_size=128, n_samples=10, save_dir="test_patches"
    )
    
    print(f"Extracted {len(patches)} test patches with shape {patches[0].shape}")
