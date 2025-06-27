import rasterio
import os

# Check CRS of your raster files
data_path = '/project/galacticbulge/ruin_repo'
raster_dir = os.path.join(data_path, 'rasters')

for filename in ['Chaco_dem.tif', 'La_plata_dem.tif', 'Tile 1.tif']:
    raster_path = os.path.join(raster_dir, filename)
    if os.path.exists(raster_path):
        with rasterio.open(raster_path) as src:
            print(f"\n{filename}:")
            print(f"  CRS: {src.crs}")
            print(f"  Units: {src.crs.linear_units if src.crs else 'Unknown'}")
            print(f"  Bounds: {src.bounds}")
            
            # Check if it's geographic (degrees) or projected (usually meters)
            if src.crs and src.crs.is_geographic:
                print(f"  → Geographic CRS (coordinates in degrees)")
            elif src.crs and src.crs.is_projected:
                print(f"  → Projected CRS (coordinates likely in meters)")
    else:
        print(f"{filename} not found")