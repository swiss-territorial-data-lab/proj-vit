import os
import rasterio
import numpy as np
from tqdm import tqdm
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import from_origin, from_bounds


ndsm_folder = '/mnt/Data2/sli/dataset/lidar/swissSURFACE3D/swiss/'
output_folder = '/mnt/Data2/sli/dataset/img/Scratch_images/tiles/'
scratch_img_folder = '/mnt/Data2/sli/dataset/img/Scratch_images/Images/2020/GE_Full/'
log_path = os.path.join(output_folder, "log.txt")

dmg_tif_ls = ['scratch_20200316_1144_12501_0_10.tif', 'scratch_20200316_1047_12501_0_6.tif']
img_ls = [img for img in os.listdir(scratch_img_folder) if img.endswith('.tif') and img not in dmg_tif_ls]



for img in tqdm(img_ls):
        
    with rasterio.open(os.path.join(scratch_img_folder, img)) as dst:
        src_img = np.array(dst.read())
        dst_bbox = dst.bounds

    long_range = range(int(dst_bbox[0] / 1000), int(dst_bbox[2] / 1000) + 1)
    lati_range = range(int(dst_bbox[1] / 1000), int(dst_bbox[3] / 1000) + 1)

    ndsm_ls = []
    ndsm_index_ls = ['{}-{}'.format(x, y) for x in long_range for y in lati_range]
    for filename in os.listdir(ndsm_folder):
        if filename.split('_')[2] in ndsm_index_ls:
            ndsm_ls.append(filename)

    if len(ndsm_ls):
        merged_ndsm, out_trans = merge([rasterio.open(os.path.join(ndsm_folder, file)) for file in ndsm_ls])
    else:
        with open(log_path, "a") as result_file:
            result_file.write('No matched nDSM tiles: {}'.format(img) + "\n")
        continue

    # Create a MemoryFile to write the ndarray data
    with rasterio.MemoryFile() as memfile:
        # Create a dataset using the MemoryFile
        with memfile.open(driver='GTiff', height=merged_ndsm.shape[1], width=merged_ndsm.shape[2], count=1, dtype=merged_ndsm.dtype, crs=dst.crs, transform=out_trans) as dataset:
            # Write the ndarray data to the dataset
            dataset.write(np.squeeze(merged_ndsm), 1)

        # Create a DatasetReader object from the MemoryFile
        merged_ndsm = rasterio.open(memfile)

    if not (
        merged_ndsm.bounds.left <= dst_bbox.left
        and merged_ndsm.bounds.bottom <= dst_bbox.bottom
        and merged_ndsm.bounds.right >= dst_bbox.right
        and merged_ndsm.bounds.top >= dst_bbox.top
        ):
        # print('Out of nDSM boundary: {}'.format(img))
        with open(log_path, "a") as result_file:
            result_file.write('Out of nDSM boundary: {}'.format(img) + "\n")
        continue

    dst_window = rasterio.windows.from_bounds(dst_bbox.left, dst_bbox.bottom, dst_bbox.right, dst_bbox.top, out_trans)
    seg_ndsm = merged_ndsm.read(window=dst_window)

    # Get metadata from the original GeoTiff
    metadata = dst.meta.copy()

    metadata.update({'count': 5})

    src_img[[0, 1, 2, 3], :, :] = src_img[[1, 2, 3, 0], :, :]
    dst_img = np.vstack([src_img, seg_ndsm])

    # Save the ndsm_subset as a new GeoTiff file
    with rasterio.open(os.path.join(output_folder, img), 'w', **metadata) as dataset:
        dataset.write(dst_img)
    

