from Mosaic_Libraries.read_path import *
from Mosaic_Libraries.cloud_detect import *
from Mosaic_Libraries.global_params import *
import numpy as np
import logging


def main_cd(prob_threshold, cloud_cover_threshold):
    ###############################
    # Global Variables/Parameters #
    ###############################

    output_files_path = "./Mosaic_Output"
    OPTICAL_BAND_NAMES = ["01", "02", "04", "05", "08", "8A", "09", "10", "11", "12"]
    # prob_threshold = 0.9
    # cloud_cover_threshold = 40

    # Load Model
    CLOUD_CLASSIFIER_MODEL = LoadModel_CloudClassifier()

    # Reading the file paths of the AOI cropped image tiles
    aoi_fps = get_fps(output_files_path, OPTICAL_BAND_NAMES, "band??.tif")

    # Calculating the dimensions of the smallest resolution image band
    ds_width, ds_height = min_dimensions(aoi_fps)

    # Downsampling all the images to the same minimum resolution and appending to a list
    Bands = band_list(aoi_fps, ds_height, ds_width, reduce_val=True)

    # Stacking all the Optical Bands
    i_stack = StackSentinelBands(Bands)

    # i_rgb = StackSentinelBands([Bands[2], Bands[1], Bands[0]])
    Is = np.array([i_stack])

    # Predicting the pixel-wise probability
    PredictedData = CloudClassfier_PixelPredict(Is, CLOUD_CLASSIFIER_MODEL)
    ClassProbs = PredictedData[0, :, :, 1]

    # Calculating the cloud cover percentage from the given output using a probability threshold
    percentage = cloud_cover(ClassProbs, prob_threshold)
    logging.info(f"Cloud Cover Percentage: {percentage} %")

    if percentage > cloud_cover_threshold:
        logging.info("Co-registration cannot be performed in this Area of Interest due to excessive cloud cover")
        tmp = query_handler(order_id, "Cloud cover percentage exceeded")
        instance_id = tmp['instance_id']
        s3_client.upload_file("./log.txt", PARAMS["outputs"]["s3_bucket"],
                              PARAMS["outputs"]["s3_path"] + order_id + '/' + 'log.txt')
        ecs = boto3.client('ecs',
                               aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                               aws_secret_access_key='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',
                               region_name='ap-south-1')
        ecs.stop_task(task=instance_id, cluster="airborne-cluster", reason="Terminated due to error")
