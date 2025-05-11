import numpy as np
import geopandas as gpd
import random
import os
from tqdm import tqdm
from shapely.geometry import box

def normalize_patch(patch):
    """Normalize patch values to 0-1 range"""
    patch_min = np.min(patch)
    patch_max = np.max(patch)
    
    if patch_max > patch_min:
        normalized = (patch - patch_min) / (patch_max - patch_min)
    else:
        normalized = np.zeros_like(patch)
    
    return normalized

def perturb_patch(patch, noise_level=0.05):
    """Apply random perturbations to a patch for augmentation"""
    # Make a copy to avoid modifying the original
    perturbed = patch.copy()
    
    # Add random noise
    noise = np.random.normal(0, noise_level, perturbed.shape)
    perturbed = perturbed + noise
    
    # Apply random brightness/contrast adjustments
    gamma = random.uniform(0.8, 1.2)
    perturbed = np.power(perturbed, gamma)
    
    # Random horizontal/vertical flip
    if random.random() > 0.5:
        perturbed = np.fliplr(perturbed)
    if random.random() > 0.5:
        perturbed = np.flipud(perturbed)
    
    # Random rotation (0, 90, 180, or 270 degrees)
    k = random.randint(0, 3)
    perturbed = np.rot90(perturbed, k)
    
    # Clip values to valid range
    perturbed = np.clip(perturbed, 0, 1)
    
    return perturbed

def reshape_for_model(patches):
    """Reshape patches to model input format"""
    # Add channel dimension if needed
    if len(patches.shape) == 3:  # (n_samples, height, width)
        patches = np.expand_dims(patches, axis=-1)
    
    # Ensure we have float data for model input
    return patches.astype(np.float32)

def create_contrastive_pairs(patches, patch_locations, sites_gdf, positive_ratio=0.3, save_dir=None):
    """
    Create positive and negative pairs for contrastive learning
    
    Args:
        patches: Array of extracted patches
        patch_locations: List of geometries representing patch locations
        sites_gdf: GeoDataFrame of known archaeological sites
        positive_ratio: Ratio of positive pairs in the dataset
        save_dir: Directory to save pairs (optional)
        
    Returns:
        X1: First elements of pairs
        X2: Second elements of pairs
        labels: 1 for positive pairs, 0 for negative pairs
    """
    # Convert patch locations to GeoDataFrame
    patches_gdf = gpd.GeoDataFrame(geometry=patch_locations)
    patches_gdf.crs = sites_gdf.crs  # Ensure same coordinate system
    
    # Find patches that intersect with known sites
    intersects = gpd.sjoin(patches_gdf, sites_gdf, predicate='intersects', how='left')
    site_patches_idx = intersects.dropna().index.tolist()
    
    # Create positive pairs from patches containing sites
    X1_positive = []
    X2_positive = []
    for idx in tqdm(site_patches_idx, desc="Creating positive pairs"):
        patch = patches[idx]
        # Normalize patch
        patch = normalize_patch(patch)
        # Create a perturbed version of the same patch
        perturbed = perturb_patch(patch)
        
        X1_positive.append(patch)
        X2_positive.append(perturbed)
    
    # Create negative pairs
    non_site_patches_idx = list(set(range(len(patches))) - set(site_patches_idx))
    n_negative_pairs = int(len(site_patches_idx) * (1 - positive_ratio) / positive_ratio)
    
    X1_negative = []
    X2_negative = []
    for _ in tqdm(range(n_negative_pairs), desc="Creating negative pairs"):
        idx1, idx2 = random.sample(non_site_patches_idx, 2)
        patch1 = normalize_patch(patches[idx1])
        patch2 = normalize_patch(patches[idx2])
        
        X1_negative.append(patch1)
        X2_negative.append(patch2)
    
    # Combine positive and negative pairs
    X1 = np.array(X1_positive + X1_negative)
    X2 = np.array(X2_positive + X2_negative)
    labels = np.concatenate([
        np.ones(len(X1_positive)),
        np.zeros(len(X1_negative))
    ])
    
    # Reshape for model input
    X1 = reshape_for_model(X1)
    X2 = reshape_for_model(X2)
    
    print(f"Created {len(X1_positive)} positive pairs and {len(X1_negative)} negative pairs")
    
    # Save pairs if directory is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X1.npy"), X1)
        np.save(os.path.join(save_dir, "X2.npy"), X2)
        np.save(os.path.join(save_dir, "labels.npy"), labels)
    
    return X1, X2, labels

def load_contrastive_pairs(save_dir):
    """Load saved contrastive pairs"""
    X1 = np.load(os.path.join(save_dir, "X1.npy"))
    X2 = np.load(os.path.join(save_dir, "X2.npy"))
    labels = np.load(os.path.join(save_dir, "labels.npy"))
    
    return X1, X2, labels

if __name__ == "__main__":
    # Test data augmentation
    from data_loader import load_shapefiles, load_site_locations, get_raster_paths, sample_random_patches
    import matplotlib.pyplot as plt
    
    # Load a single patch for testing
    base_path = '/media/usb'
    raster_paths = get_raster_paths(base_path, limit=1)
    patches, locations, _ = sample_random_patches(raster_paths, patch_size=256, n_samples=5)
    
    # Visualize original and perturbed patches
    if len(patches) > 0:
        patch = patches[0]
        if len(patch.shape) == 3 and patch.shape[0] == 1:  # Handle single-band data
            patch = patch[0]
        
        normalized = normalize_patch(patch)
        perturbed = perturb_patch(normalized)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(normalized, cmap='gray')
        ax1.set_title('Original Patch')
        ax2.imshow(perturbed, cmap='gray')
        ax2.set_title('Perturbed Patch')
        plt.savefig("augmentation_test.png")
        plt.close()
        
        print("Saved test augmentation to augmentation_test.png")