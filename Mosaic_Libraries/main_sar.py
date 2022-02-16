from Mosaic_Libraries.read_path import *
from Mosaic_Libraries.image_processing import *

import os
import logging


def main_sar(image_file_ending, image_name, json_name):
    ###############################
    # Global Variables/Parameters #
    ###############################
    image_type = "sar"

    json_file_path = "./Temp Files"
    crsystem_file_path = "./Temp Files/crsystem.txt"
    input_files_path = "./Downloaded Datasets/SAR"
    output_files_path = "./Mosaic_Output"
    reproj_files_path = "./Temp Files/SAR/Reprojected"
    cropped_files_path = "./Temp Files/SAR/Cropped"

    ##############################
    # Reading AOI and Image Data #
    ##############################

    # Reading information from JSON file
    title, intersections, coverWholeAoi = data_json(json_file_path+"/"+json_name, image_type)
    logging.info("JSON file read successfully.")

    # Transforming the coordinates of the AOI from (long,lat) to the required coordinate system
    crsystem = open(crsystem_file_path).read()
    aoi_coords = coord_transform(intersections, crsystem)

    # Reading the file paths of the image tiles
    fps = get_fps(input_files_path, title, image_file_ending)

    ##########################
    # SAR Image Reprojection #
    ##########################

    # Reprojecting the SAR images to the required coordinate system and saving the reprojected images
    for i in range(len(fps)):
        dest_path = reproj_files_path + "/reproj_" + image_name + str(i+1) + ".tif"
        sar_reproj(fps[i], dest_path, crsystem)

    logging.info("SAR image re-projection performed successfully.")

    #######################
    # Cropping AOI region #
    #######################

    # Reading the file paths of the reprojected SAR images
    sar_reproj_fps = file_search("reproj_" + image_name + "?.tif", reproj_files_path, current_only=True)

    # Create a polygon of the AOI and cropping the AOI region off each image tile
    polygon = poly_list(aoi_coords)
    crop_img, crop_trans = aoi_crop(sar_reproj_fps, polygon)
    logging.info("AOI region cropped successfully.")

    if not coverWholeAoi:
        # Saving the AOI cropped image tiles as TIF images
        for i in range(len(fps)):
            outpath = cropped_files_path + "/crop_" + image_name + str(i+1) + ".tif"
            save_image(crop_img[i], outpath, fps[0], crsystem, crop_trans[i], "Gtiff", 1)

        #######################
        # AOI Image Mosaicing #
        #######################

        # Reading the file paths of the AOI cropped image tiles
        cropped_fps = file_search("crop_" + image_name + "?.tif", cropped_files_path, True)

        # Creating a mosaic of the cropped tiles to get the final AOI
        mosaic, out_trans = stitch(cropped_fps)

        # Saving the AOI mosaic
        outpath = output_files_path + "/SAR.tif"
        save_image(mosaic, outpath, fps[0], crsystem, out_trans, "Gtiff", 1)

        logging.info("Image Mosaicing performed successfully.")

        # Optional: Deleting the saved cropped image tiles to save space
        for fp in cropped_fps:
            os.unlink(fp)

    else:
        # Saving the cropped AOI
        outpath = output_files_path + "/SAR.tif"
        save_image(crop_img[0], outpath, fps[0], crsystem, crop_trans[0], "Gtiff", 1)

    logging.info("AOI image saved successfully")
