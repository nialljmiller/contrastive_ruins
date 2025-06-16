import numpy as np
import geopandas as gpd
import random
import os
from tqdm import tqdm
from shapely.geometry import box
from scipy.ndimage import rotate, gaussian_filter

def normalize_patch(patch):
    """Normalize patch values to 0-1 range"""
    patch_min = np.min(patch)
    patch_max = np.max(patch)
    
    if patch_max > patch_min:
        normalized = (patch - patch_min) / (patch_max - patch_min)
    else:
        normalized = np.zeros_like(patch)
    
    return normalized

def apply_jitter(patch, max_offset=5):
    """Apply jitter by shifting the image slightly in x and y directions"""
    offset_x, offset_y = np.random.randint(-max_offset, max_offset+1, size=2)
    jittered = np.roll(patch, offset_x, axis=1)
    jittered = np.roll(jittered, offset_y, axis=0)
    return jittered

def apply_slight_rotation(patch, max_angle=5):
    """Apply slight rotation (<5deg)"""
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate(patch, angle, reshape=False, mode='nearest')

def apply_random_rotation(patch, max_angle=25):
    """Apply random rotation up to 25 degrees"""
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate(patch, angle, reshape=False, mode='nearest')

def apply_noise(patch, noise_level=0.05):
    """Add random noise to patch"""
    noise = np.random.normal(0, noise_level, patch.shape)
    noisy = patch + noise
    return np.clip(noisy, 0, 1)

def perturb_patch(patch, augmentation_types=None):
    """
    Apply random perturbations to a patch for augmentation
    
    Args:
        patch: Input patch to augment
        augmentation_types: List of augmentation types to apply, or None for random selection
    
    Returns:
        Perturbed patch
    """
    # Normalize input patch first if not already normalized
    if np.max(patch) > 1.0 or np.min(patch) < 0.0:
        patch = normalize_patch(patch)
    
    # Make a copy to avoid modifying the original
    perturbed = patch.copy()
    
    # If no specific augmentations requested, randomly select 2-3 to apply
    if augmentation_types is None:
        available_augmentations = [
            'jitter',          # 1. Your requested jitter
            'slight_rotate',   # 2. Your requested slight rotation (<5deg)
            'random_rotate',   # 3. Your requested 25deg rotation
            'noise',           # 4. Your requested noise
            'flip',            # Additional augmentation
            'crop',            # Additional augmentation
            'brightness',      # Additional augmentation
            'blur'             # Additional augmentation
        ]
        n_augmentations = random.randint(2, 3)
        augmentation_types = random.sample(available_augmentations, n_augmentations)
    
    # Apply selected augmentations
    for aug_type in augmentation_types:
        if aug_type == 'jitter':
            perturbed = apply_jitter(perturbed)
        
        elif aug_type == 'slight_rotate':
            perturbed = apply_slight_rotation(perturbed)
        
        elif aug_type == 'random_rotate':
            perturbed = apply_random_rotation(perturbed)
        
        elif aug_type == 'noise':
            perturbed = apply_noise(perturbed)
        
        elif aug_type == 'flip':
            # Random horizontal/vertical flip
            if random.random() > 0.5:
                perturbed = np.fliplr(perturbed)
            if random.random() > 0.5:
                perturbed = np.flipud(perturbed)
        
        elif aug_type == 'crop':
            # Random crop and resize
            h, w = perturbed.shape
            crop_ratio = random.uniform(0.8, 0.95)
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            start_h = random.randint(0, h - crop_h)
            start_w = random.randint(0, w - crop_w)
            
            # Perform crop
            cropped = perturbed[start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # Resize back to original size (simple resize using numpy)
            from scipy.ndimage import zoom
            zoom_factors = (h / crop_h, w / crop_w)
            perturbed = zoom(cropped, zoom_factors, order=1)
        
        elif aug_type == 'brightness':
            # Apply random brightness adjustment
            brightness_factor = random.uniform(0.8, 1.2)
            perturbed = perturbed * brightness_factor
            perturbed = np.clip(perturbed, 0, 1)
        
        elif aug_type == 'blur':
            # Apply gaussian blur
            sigma = random.uniform(0.5, 1.5)
            perturbed = gaussian_filter(perturbed, sigma=sigma)
    
    # Ensure values stay in valid range
    perturbed = np.clip(perturbed, 0, 1)
    
    return perturbed

def reshape_for_model(patches):
    """Reshape patches to model input format"""
    # Add channel dimension if needed
    if len(patches.shape) == 3:  # (n_samples, height, width)
        patches = np.expand_dims(patches, axis=-1)
    
    # Ensure we have float data for model input
    return patches.astype(np.float32)



# Replace create_contrastive_pairs in data_preparation.py with this self-supervised version

def create_contrastive_pairs(patches, patch_locations, sites_gdf=None, positive_ratio=0.5, save_dir=None):
    """
    Create positive and negative pairs for SELF-SUPERVISED contrastive learning
    
    For self-supervised learning:
    - Positive pairs: Same patch with different augmentations
    - Negative pairs: Different patches
    
    Args:
        patches: Array of extracted patches
        patch_locations: List of geometries (not used in self-supervised)
        sites_gdf: Not used in self-supervised learning
        positive_ratio: Ratio of positive to total pairs
        save_dir: Directory to save pairs (optional)
        
    Returns:
        X1: First elements of pairs
        X2: Second elements of pairs
        labels: 1 for positive pairs, 0 for negative pairs
    """
    print("Creating self-supervised contrastive pairs...")
    print("- Positive pairs: Same patch + different augmentations")
    print("- Negative pairs: Different patches")
    
    X1_positive = []
    X2_positive = []
    
    # Create positive pairs: same patch with different augmentations
    for idx in tqdm(range(len(patches)), desc="Creating positive pairs"):
        patch = patches[idx]
        # Normalize patch
        patch = normalize_patch(patch)
        
        # Create two different augmented versions of the same patch
        augmented1 = perturb_patch(patch, ['jitter', 'slight_rotate', 'noise'])
        augmented2 = perturb_patch(patch, ['random_rotate', 'flip', 'brightness'])
        
        X1_positive.append(augmented1)
        X2_positive.append(augmented2)
    
    # Calculate number of negative pairs based on positive_ratio
    n_positive = len(X1_positive)
    n_negative = int(n_positive * (1 - positive_ratio) / positive_ratio)
    
    X1_negative = []
    X2_negative = []
    
    # Create negative pairs: different patches
    for _ in tqdm(range(n_negative), desc="Creating negative pairs"):
        # Select two different random patches
        idx1, idx2 = random.sample(range(len(patches)), 2)
        
        patch1 = normalize_patch(patches[idx1])
        patch2 = normalize_patch(patches[idx2])
        
        # Apply augmentations to both patches
        augmented1 = perturb_patch(patch1)
        augmented2 = perturb_patch(patch2)
        
        X1_negative.append(augmented1)
        X2_negative.append(augmented2)
    
    # Combine positive and negative pairs
    X1 = np.array(X1_positive + X1_negative)
    X2 = np.array(X2_positive + X2_negative)
    labels = np.concatenate([
        np.ones(len(X1_positive)),   # 1 for positive pairs (same patch)
        np.zeros(len(X1_negative))   # 0 for negative pairs (different patches)
    ])
    
    # Reshape for model input
    X1 = reshape_for_model(X1)
    X2 = reshape_for_model(X2)
    
    print(f"Created {len(X1_positive)} positive pairs and {len(X1_negative)} negative pairs")
    print(f"Total training pairs: {len(X1)}")
    
    # Save pairs if directory is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X1.npy"), X1)
        np.save(os.path.join(save_dir, "X2.npy"), X2)
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        print(f"Saved training pairs to {save_dir}")
    
    return X1, X2, labels

def load_contrastive_pairs(save_dir):
    """Load saved contrastive pairs"""
    X1 = np.load(os.path.join(save_dir, "X1.npy"))
    X2 = np.load(os.path.join(save_dir, "X2.npy"))
    labels = np.load(os.path.join(save_dir, "labels.npy"))
    
    return X1, X2, labels

def create_geospatial_augmentation(patch):
    """
    Apply augmentations specifically designed for geospatial/DEM data
    
    Args:
        patch: Input patch (normalized)
        
    Returns:
        Augmented patch
    """
    # Choose a random geo-specific augmentation
    aug_type = random.choice(['elevation_shift', 'local_deformation', 'shadow_simulation'])
    
    if aug_type == 'elevation_shift':
        # Small random shift in elevation values (for DEM data)
        shift = np.random.uniform(-0.02, 0.02) * np.mean(patch)
        return np.clip(patch + shift, 0, 1)
    
    elif aug_type == 'local_deformation':
        # Apply small local warping to simulate terrain variations
        h, w = patch.shape
        
        # Create random displacement fields
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=15) * 3
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=15) * 3
        
        # Create mesh grid
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Apply displacement with bounds checking
        y_new = y + dy
        x_new = x + dx
        
        # Ensure indices are within bounds
        y_new = np.clip(y_new, 0, h-1)
        x_new = np.clip(x_new, 0, w-1)
        
        # Create output array
        warped = np.zeros_like(patch)
        
        # Map values through displacement field
        for i in range(h):
            for j in range(w):
                # Get displaced position
                ni, nj = int(y_new[i, j]), int(x_new[i, j])
                warped[i, j] = patch[ni, nj]
        
        return warped
    
    elif aug_type == 'shadow_simulation':
        # Simple shadow simulation (useful for archaeological features)
        # Creates directional shadows as if sun is at angle
        
        # Pick a random direction
        axis = random.randint(0, 1)
        sigma = random.uniform(1, 2)
        
        # Apply directional blur to simulate shadow
        blurred = gaussian_filter(patch, sigma=[sigma if i == axis else 0 for i in range(2)])
        
        # Mix original with shadow effect
        alpha = random.uniform(0.7, 0.9)
        return alpha * patch + (1-alpha) * blurred
    
    return patch

if __name__ == "__main__":
    # Test data augmentation
    from data_loader import load_shapefiles, load_site_locations, get_raster_paths, sample_random_patches
    import matplotlib.pyplot as plt
    
    # Load a single patch for testing
    base_path = '/project/joycelab-niall/ruin_repo/rasters'
    raster_paths = get_raster_paths(base_path, limit=1)
    patches, locations, _ = sample_random_patches(raster_paths, patch_size=256, n_samples=5)
    
    # Visualize original and augmented patches
    if len(patches) > 0:
        patch = patches[0]
        if len(patch.shape) == 3 and patch.shape[0] == 1:  # Handle single-band data
            patch = patch[0]
        
        normalized = normalize_patch(patch)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Original
        axes[0].imshow(normalized, cmap='terrain')
        axes[0].set_title('Original Patch')
        
        # Jitter
        jittered = apply_jitter(normalized)
        axes[1].imshow(jittered, cmap='terrain')
        axes[1].set_title('Jittered')
        
        # Slight rotation (<5deg)
        slight_rotated = apply_slight_rotation(normalized)
        axes[2].imshow(slight_rotated, cmap='terrain')
        axes[2].set_title('Slight Rotation (<5°)')
        
        # Random 25deg rotation
        random_rotated = apply_random_rotation(normalized)
        axes[3].imshow(random_rotated, cmap='terrain')
        axes[3].set_title('Random Rotation (±25°)')
        
        # Noise
        noisy = apply_noise(normalized)
        axes[4].imshow(noisy, cmap='terrain')
        axes[4].set_title('Noise Added')
        
        # Geo-specific augmentation
        geo_augmented = create_geospatial_augmentation(normalized)
        axes[5].imshow(geo_augmented, cmap='terrain')
        axes[5].set_title('Geospatial Augmentation')
        
        plt.tight_layout()
        plt.savefig("augmentation_test.png")
        plt.close()
        
        print("Saved test augmentations to augmentation_test.png")
