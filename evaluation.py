import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np
from shapely.geometry import box
from matplotlib.colors import LinearSegmentedColormap

def evaluate_results(detected_ruins, known_sites, buffer_distance=100):
    """
    Evaluate detection results against known sites
    
    Args:
        detected_ruins: GeoDataFrame of detected ruins
        known_sites: GeoDataFrame of known sites
        buffer_distance: Buffer distance around known sites for evaluation
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Buffer known sites to account for spatial uncertainty
    buffered_sites = known_sites.copy()
    buffered_sites['geometry'] = known_sites.geometry.buffer(buffer_distance)
    
    # Count true positives (detected ruins that intersect with known sites)
    intersects = gpd.sjoin(detected_ruins, buffered_sites, predicate='intersects', how='left')
    true_positives = len(intersects.dropna())
    
    # Calculate metrics
    precision = true_positives / len(detected_ruins) if len(detected_ruins) > 0 else 0
    recall = true_positives / len(known_sites) if len(known_sites) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': len(detected_ruins) - true_positives,
        'false_negatives': len(known_sites) - true_positives,
        'total_detected': len(detected_ruins),
        'total_known': len(known_sites)
    }
    
    return metrics

def visualize_results(detected_ruins, known_sites, study_area=None, lidar_areas=None, 
                     base_raster=None, buffer_distance=100, save_path=None):
    """
    Visualize detection results
    
    Args:
        detected_ruins: GeoDataFrame of detected ruins
        known_sites: GeoDataFrame of known sites
        study_area: GeoDataFrame of study area (optional)
        lidar_areas: GeoDataFrame of LiDAR covered areas (optional)
        base_raster: Path to raster file for background (optional)
        buffer_distance: Buffer distance for evaluation
        save_path: Path to save visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot raster as background if provided
    if base_raster:
        try:
            import rasterio
            from rasterio.plot import show
            with rasterio.open(base_raster) as src:
                show(src, ax=ax, cmap='gray', alpha=0.5)
        except Exception as e:
            print(f"Error plotting base raster: {e}")
    
    # Plot study area and LiDAR coverage if provided
    if study_area is not None:
        study_area.plot(ax=ax, alpha=0.2, color='gray', edgecolor='black', linewidth=0.5)
    
    if lidar_areas is not None:
        lidar_areas.plot(ax=ax, alpha=0.3, color='blue', edgecolor='blue', linewidth=0.5)
    
    # Plot known sites with buffer
    if known_sites is not None and len(known_sites) > 0:
        buffered_sites = known_sites.copy()
        buffered_sites['geometry'] = known_sites.geometry.buffer(buffer_distance)
        buffered_sites.plot(ax=ax, alpha=0.3, color='green', edgecolor='green')
        known_sites.plot(ax=ax, color='darkgreen', markersize=20, marker='o', label='Known Sites')
    
    # Plot detected ruins
    if detected_ruins is not None and len(detected_ruins) > 0:
        if 'anomaly_score' in detected_ruins.columns:
            # Create colormap based on anomaly scores
            norm = plt.Normalize(detected_ruins['anomaly_score'].min(), 
                                detected_ruins['anomaly_score'].max())
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFC107', '#FF5722', '#D32F2F'])
            detected_ruins.plot(ax=ax, column='anomaly_score', cmap=cmap, 
                               markersize=15, marker='x', norm=norm, 
                               legend=True, legend_kwds={'label': 'Anomaly Score'})
        else:
            detected_ruins.plot(ax=ax, color='red', markersize=15, marker='x', label='Detected Ruins')
    
    # Add legend and title
    metrics = evaluate_results(detected_ruins, known_sites, buffer_distance)
    title = (f'Archaeological Ruins Detection Results\n'
            f'Precision: {metrics["precision"]:.2f}, Recall: {metrics["recall"]:.2f}, '
            f'F1: {metrics["f1_score"]:.2f}')
    ax.set_title(title)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    
    # Add scale bar and north arrow (simplified)
    ax.text(0.02, 0.02, 'Scale bar would go here', transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.95, 0.05, 'N↑', transform=ax.transAxes, fontsize=14, 
           bbox=dict(facecolor='white', alpha=0.7))
    
    # Save or show figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return metrics

def create_results_report(metrics, save_dir="results"):
    """
    Create a text report of detection results
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_dir: Directory to save report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "detection_report.txt"), "w") as f:
        f.write("Archaeological Ruins Detection Report\n")
        f.write("===================================\n\n")
        
        f.write("Detection Metrics:\n")
        f.write(f"- Precision: {metrics['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['recall']:.4f}\n")
        f.write(f"- F1 Score: {metrics['f1_score']:.4f}\n\n")
        
        f.write("Detection Counts:\n")
        f.write(f"- Total detected ruins: {metrics['total_detected']}\n")
        f.write(f"- Known archaeological sites: {metrics['total_known']}\n")
        f.write(f"- True positives: {metrics['true_positives']}\n")
        f.write(f"- False positives: {metrics['false_positives']}\n")
        f.write(f"- False negatives: {metrics['false_negatives']}\n")
    
    print(f"Saved detection report to {os.path.join(save_dir, 'detection_report.txt')}")

if __name__ == "__main__":
    print("Evaluation module - use in main.py")
