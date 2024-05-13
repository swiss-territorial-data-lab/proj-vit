import os
import pdal
import wget
import json
import zipfile
import argparse
import whitebox
import numpy as np
import pandas as pd
from glob import glob
from osgeo import gdal
from tqdm import tqdm
from whitebox import WhiteboxTools
from multiprocessing import Pool

# whitebox.download_wbt(linux_musl=True, reset=True) 



def main():
    args = argparse.ArgumentParser()
    args.add_argument('--csv_dir', type=str, 
                      default='',
                      help='Directory to csv files.')
    args.add_argument('--out_dir', type=str, 
                      default='',
                      help='Directory to store nDSM file.')
    args.add_argument('--nproc', type=int, 
                      default=48,
                      help='Number of processes.')
    args.add_argument('--resolution', type=float, 
                      default=1.0,
                      help='Output nDSM raster resolution in meters.')
    args = args.parse_args()

    # identify the sample data directory of the package
    out_dir = args.out_dir
    num_processes = args.nproc
    resolution = args.resolution
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.mkdir(os.path.join(out_dir, 'wget'))
        os.mkdir(os.path.join(out_dir, 'resampled_las'))
        os.mkdir(os.path.join(out_dir, 'normalized_las'))
        os.mkdir(os.path.join(out_dir, 'ndsm_tif'))
        

    csv_ls = sorted(glob(os.path.abspath(os.path.join(args.csv_dir, '*.csv'))))
    assert len(csv_ls)

    for csv_path in csv_ls:
        print(f"Start to preprocess {csv_path}:\n")
        df = pd.read_csv(csv_path, header=0)
        link_ls = df.to_numpy()

        # Multi-processing to maximize CPU utilization
        # Create a pool of worker processes
        with Pool(processes=num_processes) as pool, tqdm(total=len(link_ls)) as pbar:
            # Execute the function in parallel with multiple parameters
            for _ in pool.imap_unordered(preprocess, [(link, out_dir, resolution) for link in link_ls]):
                pbar.update(1)
                
        # # single thread debug mode
        # for link in tqdm(link_ls):
        #     args = (link, out_dir)
        #     preprocess(args)
        
    # Create a WhiteboxTools instance
    wbt = WhiteboxTools()
    wbt.verbose = False
    # Set working dir
    normalized_las_folder = os.path.join(out_dir, 'normalized_las')
    wbt.set_working_dir(normalized_las_folder)
    
    # Generate nDSM
    wbt.lidar_digital_surface_model(resolution=resolution, radius=resolution/2)
    print("Completed normalized float type DSM generation! \n")
    
    tif_ls = [file for file in os.listdir(normalized_las_folder) if file.endswith('.tif')]
    # Multi-processing to convert and scale the nDSM
    print("convert and scale the nDSM to 8 bit integer: \n")
    with Pool(processes=num_processes) as pool, tqdm(total=len(tif_ls)) as pbar:
        # Execute the function in parallel with multiple parameters
        for _ in pool.imap_unordered(convert_to_8bit, [(tif_file, out_dir) for tif_file in tif_ls]):
            pbar.update(1)
            
    print(f"Normalized DSM generation finished! \n")
    print(f"Log file is saved in {os.path.join(out_dir, "log.txt")}. \n")
    print(f"Normalized DSM files are saved in {os.path.join(out_dir, 'ndsm_tif')} \n")


def preprocess(args):
    (link, out_dir, resolution) = args
    link = str(link[0])
    log_path = os.path.join(out_dir, "log.txt")
    
    try:        
        if not (link.endswith('.copc.laz') or link.endswith('.laz') or link.endswith('.las') or link.endswith('.las.zip')):
            with open(log_path, "a") as result_file:
                result_file.write('Not supported for link: {}'.format(link) + "\n")
            return   
        
        file_name = link.split('/')[-1]
        # las_name = file_name.replace('.copc.laz', '.las')
        
        las_name = file_name.split('.')[0] + '.las'
        wget_folder = os.path.join(out_dir, 'wget')
        resampled_las_folder = os.path.join(out_dir, 'resampled_las')
        resampled_las_file = os.path.join(resampled_las_folder, las_name)
        normalized_las_folder = os.path.join(out_dir, 'normalized_las')
        normalized_las_file = os.path.join(normalized_las_folder, las_name)
        
        # skip exist files if already generated
        
        if os.path.exists(normalized_las_file):
            return
        
        try:
            wget.download(link, wget_folder, bar=None)
        except Exception as e:
            with open(log_path, "a") as result_file:
                result_file.write('Download error'.format(link) + "\n")
            return  
        
        # uncompress and resample the las file 
        wget_pts = os.path.join(wget_folder, file_name)
        
        # specific config for swisstopo data whose name inside .las.zip is different after unzipped  
        if link.endswith('.zip'):
            with zipfile.ZipFile(wget_pts, 'r') as zip_ref:
                zip_ref.extractall(wget_folder)
                
            os.remove(wget_pts)
            unzip_name = file_name.split('_')[2].replace('-', '_') + '.las'
            os.rename(os.path.join(wget_folder, unzip_name) , os.path.join(wget_folder, las_name))
            wget_pts = wget_pts[:-4]
               
        
        resample_to_las(wget_pts, resampled_las_file, resolution/2)
        os.remove(wget_pts)

        # Create a WhiteboxTools instance
        wbt = WhiteboxTools()
        wbt.verbose = False

        # Run the 'height_above_ground' tool
        wbt.height_above_ground(resampled_las_file, output=normalized_las_file)

        if not os.path.exists(normalized_las_file):
            
            with open(log_path, "a") as result_file:
                result_file.write('No ground point tile: {}'.format(link) + "\n")
            return
        else:
            os.remove(resampled_las_file)
    
    except Exception as e:
        with open(log_path, "a") as result_file:
            result_file.write(f'Failed to preprocess: {link} \n')
            result_file.write(f'\t Error message: {str(e)} \n')
        return

def resample_to_las(input_file, output_las, resolution=5.0):
    
    if input_file.endswith('copc.laz'):
        reader = "readers.copc"
    elif input_file.endswith('.laz') or input_file.endswith('.las') or input_file.endswith('.las.zip'):
        reader = "readers.las"
    else:
        raise TypeError('File type is not supported. Please use las/laz/copc.laz point clouds!')
    
    # PDAL pipeline for the resampling and conversion 
    pipeline = {
        "pipeline": [
            {
                "type": reader,
                "filename": input_file
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:17]"
            },
            {
                "type":"filters.voxelcenternearestneighbor",
                "cell":resolution,
                "where":"Classification == 2",
                "where_merge":True
            },
            {
                "type":"filters.voxelcenternearestneighbor",
                "cell":resolution,
                "where":"Classification != 2",
                "where_merge":True
            },
            {
                "type": "writers.las",
                "filename": output_las
            }
        ]
    }

    # Execute the PDAL pipeline
    pdal_pipeline = pdal.Pipeline(json.dumps(pipeline))
    pdal_pipeline.execute()


def convert_to_8bit(args):

    (tif_name, out_dir) = args
    
    output_file = os.path.join(out_dir, 'ndsm_tif', tif_name)
    input_file = os.path.join(out_dir, 'normalized_las', tif_name)
    
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