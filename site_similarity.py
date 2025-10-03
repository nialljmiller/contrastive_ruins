import argparse
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point, box
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import PCA  # Optional for visualization
import torch
import matplotlib.pyplot as plt  # Optional for quick plots

# Existing imports (assume project structure)
from data_loader import sample_site_patches, get_raster_paths
from model import load_models
from feature_extraction import extract_features
from visualization import compute_embeddings  # Reuse or inline

def parse_args():
    parser = argparse.ArgumentParser(description="Site Similarity Search using Contrastive Embeddings")
    parser.add_argument('--data_path', type=str, default='/project/galacticbulge/ruin_repo',
                        help='Base path to data')
    parser.add_argument('--csv_file', type=str, default='site_locations/site_locations.csv',
                        help='Relative path to Seans CSV')
    parser.add_argument('--grid_raster', type=str, default='rasters/Tile 1.tif',
                        help='Relative path to grid 3 raster')
    parser.add_argument('--target_rasters', nargs='+', default=['rasters/Chaco_dem.tif'],
                        help='Target rasters for prediction (space-separated)')
    parser.add_argument('--output_dir', type=str, default='output/results',
                        help='Directory for outputs')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')
    parser.add_argument('--stride', type=int, default=128, help='Stride for feature extraction')
    parser.add_argument('--similarity_threshold', type=float, default=0.8,
                        help='Cosine similarity threshold')
    parser.add_argument('--use_medoid', action='store_true', help='Use medoid instead of mean for prototype')
    parser.add_argument('--buffer_distance', type=float, default=10.0,
                        help='Buffer (meters) for excluding known sites')
    parser.add_argument('--model_dir', type=str, default='models', help='Trained models directory')
    return parser.parse_args()

def load_sites_csv(csv_path, crs="EPSG:26912"):
    df = pd.read_csv(csv_path)
    geometry = [Point(x, y) for x, y in zip(df['true_easting'], df['true_northing'])]
    sites_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    print(f"Loaded {len(sites_gdf)} sites from CSV")
    return sites_gdf

def check_and_align_crs(raster_path, gdf):
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        print(f"Raster CRS: {raster_crs}")
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
        print("Reprojected sites GDF to match raster CRS")
    return gdf, raster_crs

def extract_site_embeddings(encoder, raster_path, sites_gdf, patch_size):
    site_patches, _, _ = sample_site_patches([raster_path], sites_gdf, patch_size=patch_size)
    if len(site_patches) == 0:
        raise ValueError("No site patches extracted; check coordinates/raster overlap")
    site_embeddings = compute_embeddings(encoder, site_patches)  # Assumes function exists
    print(f"Extracted {len(site_embeddings)} site embeddings")
    return site_embeddings

def compute_prototype(embeddings, use_medoid=False):
    if use_medoid:
        dist_matrix = pairwise_distances(embeddings)
        medoid_idx = np.argmin(np.sum(dist_matrix, axis=0))
        prototype = embeddings[medoid_idx]
    else:
        prototype = np.mean(embeddings, axis=0)
    print("Prototype computed")
    return prototype

def extract_target_features(encoder, target_rasters, patch_size, stride):
    target_features = []
    target_locations = []
    for raster in target_rasters:
        feats, locs, _ = extract_features(encoder, raster, patch_size=patch_size, stride=stride)
        target_features.extend(feats)
        target_locations.extend(locs)
    target_features = np.array(target_features)
    print(f"Extracted {len(target_features)} target features")
    return target_features, target_locations

def find_similar_candidates(target_features, prototype, threshold, target_locations, raster_crs):
    similarities = cosine_similarity(target_features, prototype.reshape(1, -1)).flatten()
    candidate_idx = np.where(similarities >= threshold)[0]
    candidate_geoms = [box(loc.bounds[0], loc.bounds[1], loc.bounds[2], loc.bounds[3])
                       for loc in np.array(target_locations)[candidate_idx]]
    candidates_gdf = gpd.GeoDataFrame({'similarity': similarities[candidate_idx]},
                                      geometry=candidate_geoms, crs=raster_crs)
    print(f"Identified {len(candidates_gdf)} similar candidates")
    return candidates_gdf

def filter_known_sites(candidates_gdf, sites_gdf, buffer_distance):
    buffered_sites = sites_gdf.copy()
    buffered_sites['geometry'] = buffered_sites.buffer(buffer_distance)
    joined = gpd.sjoin(candidates_gdf, buffered_sites, how='left', predicate='intersects')
    novel_candidates = joined[joined.index_right.isna()].drop(columns=['index_right'])
    print(f"Filtered to {len(novel_candidates)} novel candidates")
    return novel_candidates

def export_results(gdf, output_dir, filename='novel_sites.shp'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    gdf.to_file(output_path, driver='ESRI Shapefile')
    print(f"Exported to {output_path}")

# Optional: Quick visualization (e.g., PCA plot of embeddings)
def visualize_embeddings(site_embeddings, target_features, output_dir):
    all_emb = np.vstack([site_embeddings, target_features])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_emb)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:len(site_embeddings), 0], reduced[:len(site_embeddings), 1],
                c='blue', label='Known Sites', alpha=0.6)
    plt.scatter(reduced[len(site_embeddings):, 0], reduced[len(site_embeddings):, 1],
                c='red', label='Target Features', alpha=0.3)
    plt.legend()
    plt.title('PCA of Embeddings')
    plt.savefig(os.path.join(output_dir, 'embeddings_pca.png'))
    plt.close()

def main():
    args = parse_args()
    data_path = args.data_path
    csv_path = os.path.join(data_path, args.csv_file)
    grid_raster = os.path.join(data_path, args.grid_raster)
    target_rasters = [os.path.join(data_path, r) for r in args.target_rasters]
    output_dir = os.path.join(data_path, args.output_dir)

    # Load models
    encoder, _ = load_models(save_dir=os.path.join(data_path, args.model_dir))

    # Step 1: Load CSV
    sites_gdf = load_sites_csv(csv_path)

    for file_grid_raster in ['Tile 12.tif', 'Tile 16.tif', 'Tile 1.tif', 'Tile 3.tif', 'Tile 7.tif', 'Tile 13.tif', 'Tile 17.tif', 'Tile 20.tif', 'Tile 4.tif', 'Tile 8.tif', 'Tile 10.tif', 'Tile 14.tif', 'Tile 18.tif', 'Tile 2.tif', 'Tile 5.tif', 'Tile 9.tif', 'Tile 11.tif', 'Tile 15.tif', 'Tile 19.tif', 'Tile_3_hillshade.tif', 'Tile 6.tif']:

        grid_raster = os.path.join(data_path, file_grid_raster)

        # Step 2: Verify/align CRS
        sites_gdf, raster_crs = check_and_align_crs(grid_raster, sites_gdf)

        # Step 3: Extract site embeddings
        site_embeddings = extract_site_embeddings(encoder, grid_raster, sites_gdf, args.patch_size)

        # Step 4: Compute prototype
        prototype = compute_prototype(site_embeddings, args.use_medoid)

        # Step 5: Extract target features
        target_features, target_locations = extract_target_features(
            encoder, target_rasters, args.patch_size, args.stride
        )

        # Step 6: Find similar candidates
        candidates_gdf = find_similar_candidates(
            target_features, prototype, args.similarity_threshold, target_locations, raster_crs
        )

        # Step 7: Filter known sites
        novel_candidates = filter_known_sites(candidates_gdf, sites_gdf, args.buffer_distance)

        # Step 8: Export
        export_results(novel_candidates, output_dir)

        # Optional: Visualize
        visualize_embeddings(site_embeddings, target_features, output_dir)

if __name__ == "__main__":
    main()