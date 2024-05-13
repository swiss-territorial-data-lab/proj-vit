import os
import argparse
import rasterio
import numpy as np

from tqdm import tqdm
from rasterio.transform import from_origin

def main():
    
    args = config_parser()
    
    input_folder = args.input_folder 
    output_folder = args.output_folder
    
    # List all the .tif files in the folder
    tif_files = [file for file in os.listdir(input_folder) if file.endswith(".tif")]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each .tif file
    for tif_file in tqdm(tif_files):
        tif_file_path = os.path.join(input_folder, tif_file)

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


def config_parser():
    parser = argparse.ArgumentParser(description="Generate 5 band image tiles in given resolution.")
    parser.add_argument("--input_folder", 
                        default="~/dataset/", 
                        help="Path to 5-band imagery folder")
    parser.add_argument("--output_folder", 
                        default="~/dataset/img/tiles/", 
                        help="Path to output folder")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
	main()