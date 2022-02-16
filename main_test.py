# from Mosaic_Libraries.main_dd import *
from Mosaic_Libraries.main_optical import *
from Mosaic_Libraries.main_sar import *
from Mosaic_Libraries.main_cd import *
import logging

logging.basicConfig(filename="log.txt",
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

json_file_name = "test.json"
op_bands_name = ["band01", "band02", "band04", "band05", "band08", "band8A", "band09", "band10", "band11", "band12"]
op_bands_ending = ["*B01.jp2", "*B02.jp2", "*B04.jp2", "*B05.jp2", "*B08.jp2", "*B8A.jp2", "*B09.jp2", "*B10.jp2",
                   "*B11.jp2", "*B12.jp2"]
sar_bands_name = ["vv"] # ["vv", "vh"]
sar_bands_ending = ["*001.tiff"] # ["*001.tiff", "*002.tiff"]
prob_threshold = 0.6
cloud_cover_threshold = 60


# Downloading and extracting data from Sentinel
# main_dd(json_file_name)

# RGB AOI Mosaic
main_optical("*TCI.jp2", "RGB", json_file_name, isRGB=True)

# Optical Bands AOI Mosaic
for i in range(len(op_bands_name)):
    main_optical(op_bands_ending[i], op_bands_name[i], json_file_name, isRGB=False)

# SAR Bands AOI Mosaic
for i in range(len(sar_bands_name)):
    main_sar(sar_bands_ending[i], sar_bands_name[i], json_file_name)

# Cloud Detection Algorithm
main_cd(prob_threshold, cloud_cover_threshold)
