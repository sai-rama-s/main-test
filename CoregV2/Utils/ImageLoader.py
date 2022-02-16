'''
Image Loader
'''

# Imports
import os
import cv2
import glob
import numpy as np
from .ImageUtils import DEFAULT_OPTIONS
import logging
import rasterio
import boto3
# from rasterio.plot import show
import matplotlib.pyplot as plt
import requests
from dataclasses import dataclass, asdict

@dataclass
class Client:
    url: str
    headers: dict

    def run_query(self, query: str, variables: dict, extract=False):
        request = requests.post(
            self.url,
            headers=self.headers,
            json={"query": query, "variables": variables},
        )
        assert request.ok, f"Failed with code {request.status_code}"
        return request.json()

    update_status = lambda self, orderId, status: self.run_query(
        """
            mutation MyMutation($_id: uuid!, $_status: String) {
					  update_order_details_by_pk(pk_columns: {id: $_id}, _set: {status: $_status}) {
				    id
				    status
  }
}
        """,
        { "_id": orderId, "_status": status},
    )

HASURA_URL = "https://galaxeye-airborne.hasura.app/v1/graphql"
HASURA_HEADERS = {"X-Hasura-Admin-Secret": 'ex2IRh1w1b3ikgYBao8GuFHhsMmGKwm10p1M6wB2mFm86p44wQ0QVOjdmplKli2s'}

client = Client(url=HASURA_URL, headers=HASURA_HEADERS)

def query_handler(orderID, status):
    user_response = client.update_status(orderID, status)
    if user_response.get("errors"):
        return {"message": user_response["errors"][0]["message"]}, 400
    else:
        user = user_response["data"]["update_order_details_by_pk"]
        return user


# Main Vars
DIRPATHS = {
    "Noisy": "SN6/Noisy/",
    "Denoised": "SN6/Denoised/",
    "Sentinel": "Sentinel/",
    "Other": "Other/"
}
SUBDIRS_OPTICAL = {
    "Noisy": "",
    "Denoised": "PS-RGBNIR/",
    "Sentinel": "PS-RGBNIR/",
    "Other": "PS-RGBNIR/"
}
SUBDIRS_SAR = {
    "Noisy": "",
    "Denoised": "sar/",
    "Sentinel": "sar/",
    "Other": "sar/"
}

logging.basicConfig(filename = "log.txt",
                    filemode = 'a',
                    level=logging.DEBUG,
                    format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt = '%Y-%m-%d %H:%M:%S')

# Modes
# SN6_MODE_OPTICAL = "PS-RGBNIR"
# SN6_MODE_SAR = "SAR-Intensity"

# Prefixes and Suffixes
NAME_OPTICAL = {
    "Noisy": {
        "prefix": f"SN6_Train_AOI_11_Rotterdam_PS-RGBNIR_",
        "suffix": "",
        "ext": ".tif"
    },
    "Denoised": {
        "prefix": f"SN6_Train_AOI_11_Rotterdam_PS-RGBNIR_",
        "suffix": "",
        "ext": ".tif"
    },
    "Sentinel": {
        "prefix": f"Sentinel_",
        "suffix": "_Optical",
        "ext": ".tiff"
    },
    "Other": {
        "prefix": "Other_",
        "suffix": "_Optical",
        "ext": ".*"
    }
}

NAME_SAR = {
    "Noisy": {
        "prefix": f"SN6_Train_AOI_11_Rotterdam_SAR-Intensity_",
        "suffix": "_Spk",
        "ext": ".tif"
    },
    "Denoised": {
        "prefix": f"SN6_Train_AOI_11_Rotterdam_SAR-Intensity_",
        "suffix": "_Spk",
        "ext": ".tif"
    },
    "Sentinel": {
        "prefix": f"Sentinel_",
        "suffix": "_SAR",
        "ext": ".tiff"
    },
    "Other": {
        "prefix": "Other_",
        "suffix": "_SAR",
        "ext": ".*"
    }
}

# Main Functions
# Search Functions
def LoadImagePaths(dir_path, optical_subdir=SUBDIRS_OPTICAL["Noisy"], sar_subdir=SUBDIRS_SAR["Noisy"], optical_name=NAME_OPTICAL["Noisy"], sar_name=NAME_SAR["Noisy"]):
    Optical_paths = glob.glob(os.path.join(dir_path, optical_subdir, '*' + optical_name["ext"]))
    SAR_paths = glob.glob(os.path.join(dir_path, sar_subdir, '*' + sar_name["ext"]))

    Optical_paths.sort()
    SAR_paths.sort()

    Optical_pathSplits = [os.path.splitext(os.path.basename(path)) for path in Optical_paths]
    SAR_pathSplits = [os.path.splitext(os.path.basename(path)) for path in SAR_paths]
    Optical_names = [ps[0] for ps in Optical_pathSplits]
    Optical_exts = [ps[1] for ps in Optical_pathSplits]
    SAR_names = [ps[0] for ps in SAR_pathSplits]
    SAR_exts = [ps[1] for ps in SAR_pathSplits]

    Optical_ids = []
    Optical_exts_cleaned = []
    for i in range(len(Optical_names)):
        name = Optical_names[i]
        ext = Optical_exts[i]
        optical_start = optical_name["prefix"]
        optical_end = optical_name["suffix"]
        if name.startswith(optical_start) and name.endswith(optical_end):
            name = name[len(optical_start) : len(name) - len(optical_end)]
            Optical_ids.append(name)
            Optical_exts_cleaned.append(ext)

    SAR_ids = []
    SAR_exts_cleaned = []
    for i in range(len(SAR_names)):
        name = SAR_names[i]
        ext = SAR_exts[i]
        sar_start = sar_name["prefix"]
        sar_end = sar_name["suffix"]
        if name.startswith(sar_start) and name.endswith(sar_end):
            name = name[len(sar_start) : len(name) - len(sar_end)]
            SAR_ids.append(name)
            SAR_exts_cleaned.append(ext)

    # Find Common IDs
    Optical_ids_set = set(Optical_ids)
    SAR_ids_set = set(SAR_ids)
    Common_ids = list(Optical_ids_set.intersection(SAR_ids_set))
    Common_ids.sort()

    # Return Common Paths
    Common_Paths = []
    for i in range(len(Common_ids)):
        opt_ext = Optical_exts_cleaned[Optical_ids.index(Common_ids[i])]
        opt_path = dir_path + optical_subdir + optical_name["prefix"] + Common_ids[i] + optical_name["suffix"] + opt_ext
        sar_ext = SAR_exts_cleaned[SAR_ids.index(Common_ids[i])]
        sar_path = dir_path + sar_subdir + sar_name["prefix"] + Common_ids[i] + sar_name["suffix"] + sar_ext

        Common_Paths.append([opt_path, sar_path])

    return Common_Paths

# Path Functions
def GetFilePath(image_index, mode_name, dir_path=DIRPATHS["Noisy"], subdir=SUBDIRS_OPTICAL["Noisy"], TIFF_FILE_IDS=[]):
    image_id = TIFF_FILE_IDS[image_index]
    return f'{dir_path}{subdir}{mode_name["prefix"]}{image_id}{mode_name["suffix"]}{mode_name["ext"]}'

# Load Functions
def GET_FILE_PATHS():
    # Load All Images
    FILE_PATHS = {}
    for k in DIRPATHS.keys():
        FILE_PATHS[k] = LoadImagePaths(DIRPATHS[k], optical_subdir=SUBDIRS_OPTICAL[k], sar_subdir=SUBDIRS_SAR[k], optical_name=NAME_OPTICAL[k], sar_name=NAME_SAR[k])
        print("Available " + k + " Images:", len(FILE_PATHS[k]))

    return FILE_PATHS

def LoadBands(path_optical, path_sar, options=DEFAULT_OPTIONS):
    # Load Bands
    # Load Optical Bands
    I_1_bands = None
    ext_1 = os.path.splitext(path_optical)[1]
    if ext_1 in [".tif", ".tiff"]:
        try:
            I_1_bands = rasterio.open(path_optical).read().astype('float32')
        except:
            if os.environ['ORDER_ID'] is not None:
                order_id = os.environ['ORDER_ID']
            else:                       
                order_id = "trial"
            tmp = query_handler(order_id, "Error")
            instance_id = tmp['instance_id']
            s3_client.upload_file("./log.txt", PARAMS["outputs"]["s3_bucket"],
                                  PARAMS["outputs"]["s3_path"] + order_id + '/' + 'log.txt')
            ecs = boto3.client('ecs',
                               aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                               aws_secret_access_key='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',
                               region_name='ap-south-1')
            ecs.stop_task(task=instance_id, cluster="airborne-cluster", reason="Terminated due to error")
            
        I_1_bands = I_1_bands[-1]
    else:
        try:
            I_1_bands = cv2.imread(path_optical)
        except:
            if os.environ['ORDER_ID'] is not None:
                order_id = os.environ['ORDER_ID']
            else:                       
                order_id = "trial"
            tmp = query_handler(order_id, "Error")
            instance_id = tmp['instance_id']
            s3_client.upload_file("./log.txt", PARAMS["outputs"]["s3_bucket"],
                                  PARAMS["outputs"]["s3_path"] + order_id + '/' + 'log.txt')
            ecs = boto3.client('ecs',
                               aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                               aws_secret_access_key='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',
                               region_name='ap-south-1')
            ecs.stop_task(task=instance_id, cluster="airborne-cluster", reason="Terminated due to error")
        if I_1_bands.ndim == 3:
            I_1_bands = np.mean(I_1_bands, axis=2)
        # I_1_bands_gray = np.mean(I_1_bands, axis=2)
        # I_1_bands = np.array([I_1_bands[:, :, 0], I_1_bands[:, :, 1], I_1_bands[:, :, 2], I_1_bands_gray], dtype=np.float32)
    # Load SAR Bands
    I_2_bands = None
    ext_2 = os.path.splitext(path_sar)[1]
    if ext_2 in [".tif", ".tiff"]:
        try:
            I_2_bands = rasterio.open(path_sar).read().astype('float32')
        except:
            if os.environ['ORDER_ID'] is not None:
                order_id = os.environ['ORDER_ID']
            else:                       
                order_id = "trial"
            tmp = query_handler(order_id, "Error")
            instance_id = tmp['instance_id']
            s3_client.upload_file("./log.txt", PARAMS["outputs"]["s3_bucket"],
                                  PARAMS["outputs"]["s3_path"] + order_id + '/' + 'log.txt')
            ecs = boto3.client('ecs',
                               aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                               aws_secret_access_key='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',
                               region_name='ap-south-1')
            ecs.stop_task(task=instance_id, cluster="airborne-cluster", reason="Terminated due to error")
            
        I_2_bands = I_2_bands[-1]
    else:
        try:
            I_2_bands = cv2.imread(path_sar)
        except:
            if os.environ['ORDER_ID'] is not None:
                order_id = os.environ['ORDER_ID']
            else:                       
                order_id = "trial"
            tmp = query_handler(order_id, "Error")
            instance_id = tmp['instance_id']
            s3_client.upload_file("./log.txt", PARAMS["outputs"]["s3_bucket"],
                                  PARAMS["outputs"]["s3_path"] + order_id + '/' + 'log.txt')
            ecs = boto3.client('ecs',
                               aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                               aws_secret_access_key='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',
                               region_name='ap-south-1')
            ecs.stop_task(task=instance_id, cluster="airborne-cluster", reason="Terminated due to error")
        if I_2_bands.ndim == 3:
            I_2_bands = np.mean(I_2_bands, axis=2)

    # # Check if Optical has only 1 band
    # if I_1_bands.shape[0] == 1:
    #     I_1_bands = np.array([I_1_bands[0], I_1_bands[0], I_1_bands[0], I_1_bands[0]], dtype=np.float32)

    # Display
    if options["verbose"]:
        print("Optical Bands:", I_1_bands.shape, I_1_bands.min(), I_1_bands.max())
        print("SAR Bands:", I_2_bands.shape, I_2_bands.min(), I_2_bands.max())

    if options["plot"] or options["save"]:
        # Initial Clear
        plt.clf()

        plt.figure(figsize=(20, 20))
        # Optical
        plt.subplot(1, 2, 1)
        # show(I_1_bands[-1, :, :], vmin=I_1_bands.min(), vmax=I_1_bands.max())
        # plt.imshow(I_1_bands[-1, :, :], vmin=I_1_bands.min(), vmax=I_1_bands.max())
        plt.imshow(I_1_bands, vmin=I_1_bands.min(), vmax=I_1_bands.max())
        # SAR
        plt.subplot(1, 2, 2)
        # show(I_2_bands, vmin=I_2_bands.min(), vmax=I_2_bands.max())
        plt.imshow(I_2_bands, vmin=I_2_bands.min(), vmax=I_2_bands.max())

        if options["save"]:
            plt.savefig(options["path"].format("Bands"))
        if options["plot"]:
            plt.show()

        # Final Clear
        plt.clf()

    return I_1_bands, I_2_bands

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Image Loader!")