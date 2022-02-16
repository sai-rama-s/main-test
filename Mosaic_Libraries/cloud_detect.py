import numpy as np
import os
import rasterio
from lightgbm import Booster

import s2cloudless
from s2cloudless import PixelClassifier
from s2cloudless.cloud_detector import MODEL_FILENAME
from rasterio.enums import Resampling
from Mosaic_Libraries.global_params import *


# Main Functions


def min_dimensions(fp_list):
    try:
        width_list = []
        height_list = []

        for fp in fp_list:
            f = rasterio.open(fp)
            width_list.append(f.width)
            height_list.append(f.height)

        min_width = np.array(width_list).min()
        min_height = np.array(height_list).min()

        return min_width, min_height
    except:
        error_call()


def downsample(filepath, ds_height, ds_width):
    try:
        with rasterio.open(filepath) as dataset:
            # resample data to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    ds_height,
                    ds_width
                ),
                resampling=Resampling.bilinear
            )

        return data
    except:
        error_call()


def band_list(fp_list, ds_height, ds_width, reduce_val):
    Bands = []
    for fp in fp_list:
        f = rasterio.open(fp)

        if f.height > ds_height:
            band = downsample(fp, ds_height, ds_width)[0, :, :]
        else:
            band = f.read(1)

        if reduce_val:
            band = band / 10000.0

        # print(band.shape, band.min(), band.max())
        Bands.append(band)

    return Bands


def StackSentinelBands(Bands):
    I = np.dstack(tuple(Bands))
    return I


def LoadModel_CloudClassifier():
    package_path = os.path.dirname(s2cloudless.__file__)
    model_path = os.path.join(package_path, 'models', MODEL_FILENAME)
    booster = Booster(model_file=model_path)
    classifier = PixelClassifier(booster)
    return classifier


def CloudClassfier_Predict(I, CLOUD_CLASSIFIER_MODEL):
    return CLOUD_CLASSIFIER_MODEL.image_predict(I)


def CloudClassfier_PixelPredict(I, CLOUD_CLASSIFIER_MODEL):
    return CLOUD_CLASSIFIER_MODEL.image_predict_proba(I)


# def cloud_cover(Probs, threshold):
#
#     Probs[Probs >= threshold] = 1
#     Probs[Probs < threshold] = 0
#     perc_cover = Probs.sum() * 100 / (Probs.shape[0] * Probs.shape[1])
#     return perc_cover

def cloud_cover(Probs, threshold):
    count = 0
    for i in range(Probs.shape[0]):
        for j in range(Probs.shape[1]):
            if Probs[i, j] >= threshold:
                count += 1

    perc_cover = count*100/(Probs.shape[0]*Probs.shape[1])
    return perc_cover
