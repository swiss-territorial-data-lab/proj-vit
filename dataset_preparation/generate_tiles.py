import os
import rasterio
from rasterio.transform import from_origin
import numpy as np
from tqdm import tqdm

def segment_and_save_tiles(folder_path, output_folder):
    # List all the .tif files in the folder
    tif_files = [file for file in os.listdir(folder_path) if file.endswith(".tif")]

    # Process each .tif file
    for tif_file in tqdm(tif_files):
        tif_file_path = os.path.join(folder_path, tif_file)

        # Read the original TIFF image
        with rasterio.open(tif_file_path) as src:
            image = src.read()
            transform = src.transform
            height, width = src.shape

        # Get image shape and calculate the number of tiles in both dimensions
        tile_size = 512
        nodata_value = 255
        num_tiles_height = height // tile_size
        num_tiles_width = width // tile_size

        # Loop through each tile
        for i in range(num_tiles_height):
            for j in range(num_tiles_width):
                # Extract the tile from the original image
                window = rasterio.windows.Window(j * tile_size, i * tile_size, tile_size, tile_size)
                tile = image[:, window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
                
                
                # Check if the tile is smaller than the desired size, if so, skip it
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    continue
                
                nodata_pixels = np.sum(np.all(tile[:4, :, :] == nodata_value, axis=0))
                total_pixels = tile.shape[1] * tile.shape[2]

                # Check if the tile are mostly nodata area
                if nodata_pixels / total_pixels > 0.1:
                    continue
                
                # Calculate the new transform for the tile
                tile_transform = from_origin(
                transform.xoff + window.col_off * transform.a,
                transform.yoff + window.row_off * transform.e,
                transform.a,
                -transform.e
                )
                
                # Save the tile with a new name
                tile_name = f"{os.path.splitext(os.path.basename(tif_file_path))[0]}_tile_{i * num_tiles_width + j}.tif"
                tile_path = os.path.join(output_folder, tile_name)
                with rasterio.open(
                    tile_path,
                    'w',
                    driver='GTiff',
                    height=tile_size,
                    width=tile_size,
                    count=5,
                    dtype='uint8',
                    crs=src.crs,
                    transform=tile_transform) as dst:
                    
                    dst.write(tile)

if __name__ == "__main__":
    # Specify the folder path containing the .tif files
    # folder_path = "/mnt/Data2/sli/dataset/img/italy/rgbin/"
    folder_path = "/mnt/Data2/sli/dataset/img/Scratch_images/agg_images/"
    # Specify the output folder for the segmented tiles
    # output_folder = "/mnt/Data2/sli/dataset/pretrain_ready/italy/"
    output_folder = "/mnt/Data2/sli/dataset/pretrain_ready/geneva/"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Call the combined function
    segment_and_save_tiles(folder_path, output_folder)
