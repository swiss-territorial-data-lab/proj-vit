import os
import argparse
import numpy as np
import wget
import zipfile
import pandas as pd
from glob import glob
from osgeo import gdal
from tqdm import tqdm
from whitebox import WhiteboxTools
from multiprocessing import Pool


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--csv_dir', type=str, 
                      default='/mnt/Data2/sli/dataset/lidar/swissSURFACE3D/',
                      help='Directory to csv files.')
    args.add_argument('--out_dir', type=str, 
                      default='/mnt/Data2/sli/dataset/lidar/swissSURFACE3D/',
                      help='Directory to store nDSM file.')
    args.add_argument('--nproc', type=int, 
                      default=48,
                      help='Number of processes.')
    args = args.parse_args()

    # identify the sample data directory of the package
    out_dir = args.out_dir
    num_processes = args.nproc
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.mkdir(os.path.join(out_dir, 'swissSURFACE3D'))
        os.mkdir(os.path.join(out_dir, 'no_ground_tiles'))


    csv_ls = sorted(glob(os.path.abspath(os.path.join(args.csv_dir, '*.csv'))))
    assert len(csv_ls)

    for csv_path in csv_ls:
        df = pd.read_csv(csv_path, header=None)
        link_ls = df.to_numpy()

        # Create a pool of worker processes
        with Pool(processes=num_processes) as pool, tqdm(total=len(link_ls)) as pbar:
            # Execute the function in parallel with multiple parameters
            for _ in pool.imap_unordered(calculate_ndsm, [(link, out_dir) for link in link_ls]):
                pbar.update(1)
    
    os.rmdir(os.path.join(out_dir, 'swissSURFACE3D'))


def calculate_ndsm(args):
    (link, out_dir) = args

    link = str(link[0])
    file_name = link.split('/')[-1]

    lidar_path = os.path.join(out_dir, 'swissSURFACE3D')
    wget.download(link, lidar_path, bar=None)

    if link.endswith('.zip'):
        with zipfile.ZipFile(os.path.join(lidar_path, file_name), 'r') as zip_ref:
            zip_ref.extractall(lidar_path)
        os.remove(os.path.join(lidar_path, file_name))
        file_name = file_name[:-4]
        las_name = file_name.split('_')[2].replace('-', '_') + '.las'
        os.rename(os.path.join(lidar_path, las_name) , os.path.join(lidar_path, file_name))
        
        
    # Create a WhiteboxTools instance
    wbt = WhiteboxTools()
    wbt.verbose = False

    # Output file name
    nDSM_lidar = os.path.join(lidar_path, file_name.replace('.las', '_height_above_ground.las'))

    # Run the 'height_above_ground' tool
    wbt.height_above_ground(os.path.join(lidar_path, file_name), output=nDSM_lidar)

    if not os.path.exists(nDSM_lidar):
        os.rename(os.path.join(lidar_path, file_name), os.path.join(out_dir, 'no_ground_tiles', file_name))
        return

    # Generate nDSM
    ndsm_32bit = os.path.join(out_dir, file_name.replace('.las', '_32.tif'))
    ndsm_8bit = os.path.join(out_dir, file_name.replace('.las', '.tif'))
    wbt.lidar_digital_surface_model(i=nDSM_lidar, output=ndsm_32bit, resolution=0.1, radius=0.5)

    convert_to_8bit(ndsm_32bit, ndsm_8bit)
    # Delete height las
    os.remove(nDSM_lidar)
    os.remove(ndsm_32bit)
    os.remove(os.path.join(lidar_path, file_name))


    
def convert_to_8bit(input_file, output_file):
    # Open the input TIFF file
    input_dataset = gdal.Open(input_file)

    if not input_dataset:
        print("Error: Could not open the input file.")
        return

    # Read the data from the input file
    band = input_dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    # Convert to 8-bit unsigned integer
    data[data < 0] = 0

    # safety check
    if np.median(data) > 51:
        print("Error: Elevation value is suspicious!")
        return
    
    data[data > 51] = 51
    scale_factor = 5
    scaled_data = (data * scale_factor).astype(np.uint8)

    # Create the output TIFF file
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_file, input_dataset.RasterXSize, input_dataset.RasterYSize, 1, gdal.GDT_Byte)

    if not output_dataset:
        print("Error: Could not create the output file.")
        return

    # Write the 8-bit data to the output file
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(scaled_data)

    # Copy the georeferencing information
    output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    output_dataset.SetProjection(input_dataset.GetProjection())

    # Close the datasets
    input_dataset = None
    output_dataset = None


if __name__ == '__main__':
	main()







