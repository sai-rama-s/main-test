from Mosaic_Libraries.read_path import file_search
from Mosaic_Libraries.data_download import *

import os
import logging


def main_dd(json_name):
    # Global Variables/Parameters
    json_file_path = "./Temp Files"
    op_input_files_path = "./Downloaded Datasets/Optical"
    sar_input_files_path = "./Downloaded Datasets/SAR"
    user_id = "suyashaoc"
    pwd = "GalaxEye@01"

    # Reading the product id's of the Optical ans SAR tiles from the JSON file
    op_ids, sar_ids = ids_json(json_file_path+"/"+json_name)

    # Downloading the Optical Image Data from Sentinel
    download_data(user_id, pwd, op_ids, op_input_files_path)
    print("Optical Image data downloaded successfully.")
    logging.info("Optical Image data downloaded successfully.")

    # Downloading the SAR Image Data from Sentinel
    download_data(user_id, pwd, sar_ids, sar_input_files_path)
    print("SAR Image data downloaded successfully.")
    logging.info("SAR Image data downloaded successfully.")

    # Obtaining the filepaths of the downloaded ZIP files and extracting the data
    # For Optical Image Data:
    op_zip_fps = file_search("*.zip", op_input_files_path, current_only=True)
    unzip_files(op_zip_fps)

    # For SAR Image Data:
    sar_zip_fps = file_search("*.zip", sar_input_files_path, current_only=True)
    unzip_files(sar_zip_fps)

    # Optional: Deleting the ZIP files after extraction
    # For Optical Image Data:
    for zip_fp in op_zip_fps:
        os.unlink(zip_fp)

    # For SAR Image Data:
    for zip_fp in sar_zip_fps:
        os.unlink(zip_fp)
