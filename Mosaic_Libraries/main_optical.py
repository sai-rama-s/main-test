from Mosaic_Libraries.read_path import *
from Mosaic_Libraries.image_processing import *
from Mosaic_Libraries.global_params import *

import os
import logging


def main_optical(image_file_ending, image_name, json_name, isRGB=False):
    ###############################
    # Global Variables/Parameters #
    ###############################
    image_type = "optical"
    # image_file_ending = "*B03.jp2"
    # image_name = "band03"
    # isRGB = False

    json_file_path = "./Temp Files"
    crsystem_file_path = "./Temp Files/crsystem.txt"
    input_files_path = "./Downloaded Datasets/Optical"
    output_files_path = "./Mosaic_Output"
    cropped_files_path = "./Temp Files/Optical/Cropped"

    ##############################
    # Reading AOI and Image Data #
    ##############################

    # Reading information from JSON file
    title, intersections, coverWholeAoi = data_json(json_file_path+"/"+json_name, image_type)
    logging.info("JSON file read successfully.")

    # Reading the file paths of the image tiles
    fps = get_fps(input_files_path, title, image_file_ending)

    # Reading the coordinate system of the georeferenced TIF images
    crsystem = coordsys(fps, crsystem_file_path)

    # Transforming the coordinates of the AOI from (long,lat) to the required coordinate system
    aoi_coords = coord_transform(intersections, crsystem)

    #######################
    # Cropping AOI region #
    #######################

    # Create a polygon of the AOI and cropping the AOI region off each image tile
    polygon = poly_list(aoi_coords)
    crop_img, crop_trans = aoi_crop(fps, polygon)
    logging.info("AOI region cropped successfully.")

    if not coverWholeAoi:
        # Saving the AOI cropped image tiles as TIF images
        for i in range(len(fps)):
            outpath = cropped_files_path + "/crop_" + image_name + "_" + str(i+1) + ".tif"
            if isRGB:
                save_image(crop_img[i], outpath, fps[0], crsystem, crop_trans[i], "Gtiff", 3)
            else:
                save_image(crop_img[i], outpath, fps[0], crsystem, crop_trans[i], "Gtiff", 1)

        #######################
        # AOI Image Mosaicing #
        #######################

        # Reading the file paths of the AOI cropped image tiles
        cropped_fps = file_search("crop_" + image_name + "_" + "?.tif", cropped_files_path, True)

        # Creating a mosaic of the cropped tiles to get the final AOI
        mosaic, out_trans = stitch(cropped_fps)

        # Saving the AOI mosaic
        outpath = output_files_path + "/" + image_name + ".tif"
        if isRGB:
            save_image(mosaic, outpath, fps[0], crsystem, out_trans, "Gtiff", 3)
        else:
            save_image(mosaic, outpath, fps[0], crsystem, out_trans, "Gtiff", 1)

        logging.info("Image Mosaicing performed successfully.")

        # Optional: Deleting the saved cropped image tiles to save space
        for fp in cropped_fps:
            os.unlink(fp)

    else:
        # Saving the cropped AOI
        outpath = output_files_path + "/" + image_name + ".tif"
        if isRGB:
            save_image(crop_img[0], outpath, fps[0], crsystem, crop_trans[0], "Gtiff", 3)
        else:
            save_image(crop_img[0], outpath, fps[0], crsystem, crop_trans[0], "Gtiff", 1)

    logging.info("AOI image saved successfully")
