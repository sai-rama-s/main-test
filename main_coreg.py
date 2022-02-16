'''
Main Runner for Co-Registration V2
'''

# Imports
import os
import sys
import json
import time
import boto3
import shutil
import functools
import logging
from CoregV2.utils import *
from CoregV2.ScaleSpace import *
from CoregV2.KeypointDetection import *
from CoregV2.DescriptorConstruction import *
from CoregV2.ImageRegistration import *
from CoregV2.Evaluation import *
import requests
from dataclasses import dataclass, asdict
#from CoregV2.Utils.updatestatus import query_handler

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


# Main Functions
# OS Functions
def RefreshOutputDirectories(outputs_dir, outputs_zip, optical_dir, sar_dir):
    '''
    Refreshes the output directories.
   
    '''
    # Delete outputs_dir
    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)
   
    # Delete outputs_zip
    if os.path.exists(outputs_zip):
        shutil.rmtree(outputs_zip)

    # Create outputs_dir
    os.makedirs(outputs_dir)
    
    # Create outputs_zipdir
    os.makedirs(outputs_zip)
    
    #Create Optical_dir
    os.makedirs(optical_dir)
    
    #Create Sar_dir
    os.makedirs(sar_dir)

    # Create subdirs
    #for subdir in subdirs:
     #   os.makedirs(os.path.join(outputs_dir, subdir))

# Parameterisation Functions
def GetParameterisedNormFunc(NormFuncName, AllParams):
    '''
    Applies parameters to the NormFunc.
    '''
    if NormFuncName == "Adaptive_Histogram":
        NormFunc = functools.partial(NORMALISERS[NormFuncName], 
                                            clip_limit=AllParams["clip_limit"])
    elif NormFuncName == "MinMax_GaussCorrection":
        NormFunc = functools.partial(NORMALISERS[NormFuncName], 
                                            c=AllParams["c"], gamma=AllParams["gamma"])
    elif NormFuncName == "Histogram_Matching":
        NormFunc = functools.partial(NORMALISERS[NormFuncName], 
                                            ref=AllParams["ref"])
    elif NormFuncName == "ReScale":
        NormFunc = functools.partial(NORMALISERS[NormFuncName], 
                                            minVal=AllParams["minVal"], maxVal=AllParams["maxVal"])
    else:
        NormFunc = NORMALISERS[NormFuncName]

    return NormFunc

def GetParameterisedDenoiserFunc(DenoiserFuncName, AllParams):
    '''
    Applies parameters to the DenoiserFunc.
    '''
    if DenoiserFuncName == "Adaptive_Histogram":
        DenoiserFunc = functools.partial(DENOISERS[DenoiserFuncName], AllParams["sigma_psd"])
    else:
        DenoiserFunc = DENOISERS[DenoiserFuncName]

    return DenoiserFunc

# Shift Functions
def CalculateShift(ShiftMethod, ShiftParams, locs_1, locs_2, matchIndices, options=DEFAULT_OPTIONS):
    '''
    Calculates the overall shift between two images.
    '''
    # Mean Shift Method
    if ShiftMethod == "MeanShift_Weighted_InverseDistance":
        OverallShift = SHIFT_COMPUTERS[ShiftMethod](locs_1, locs_2, 
                        matchIndices, options=options, distances=ShiftParams["matchDistances"])
    # Histogram Top K Method
    elif ShiftMethod == "HistogramShift_TopKDirections":
        OverallShift = SHIFT_COMPUTERS[ShiftMethod](locs_1, locs_2, 
                        matchIndices, options=options, distances=ShiftParams["matchDistances"], **ShiftParams[ShiftMethod])
    # Homography Method
    else:
        OverallShift = SHIFT_COMPUTERS[ShiftMethod](locs_1, locs_2, 
                        matchIndices, options=options)

    return OverallShift

    
# Runner Functions
def RunCoRegV2(PARAMS):
    '''
    Co-Registration V2 Runner
    '''
    ###############################################################################################################
    # SETUP #######################################################################################################
    ###############################################################################################################
    #Create a log file
    logging.basicConfig(filename = "log.txt",
                    level=logging.DEBUG,
                    filemode = 'a',
                    format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt = '%Y-%m-%d %H:%M:%S')
    
        
    # Connect to S3
    s3_client = boto3.client('s3',
            aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
            aws_secret_access_key='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'
    )
    # Upload to S3
    logging.info("S3 connection successfull")
    
   

    
    # Update Output Dir
    if PARAMS["options"]["DEFAULT_OPTIONS_COMMON"]["verbose_main"]:
        print("Optical Path:", PARAMS["inputs"]["optical_path"])
        print("SAR Path:", PARAMS["inputs"]["sar_path"])
        print("Output Path:", PARAMS["outputs"]["output_path"])

    # Load Options
    DEFAULT_OPTIONS_COMMON = PARAMS["options"]["DEFAULT_OPTIONS_COMMON"]
    DEFAULT_OPTIONS_OPTICAL = PARAMS["options"]["DEFAULT_OPTIONS_OPTICAL"]
    DEFAULT_OPTIONS_SAR = PARAMS["options"]["DEFAULT_OPTIONS_SAR"]
    # Update save path with dir
    DEFAULT_OPTIONS_COMMON["path"] = PARAMS["outputs"]["output_path"] 
    DEFAULT_OPTIONS_OPTICAL["path"] = PARAMS["outputs"]["output_path"] 
    DEFAULT_OPTIONS_SAR["path"] = PARAMS["outputs"]["output_path"] 

    # Copy Options
    options_common = dict(DEFAULT_OPTIONS_COMMON)
    options_optical = dict(DEFAULT_OPTIONS_OPTICAL)
    options_sar = dict(DEFAULT_OPTIONS_SAR)

    # Refresh Output Directories if saving
    if options_common["save"] or options_optical["save"] or options_sar["save"]:
        SubDirs = [
            options_common["dir"],
            options_optical["dir"],
            options_sar["dir"]
        ]
    RefreshOutputDirectories(PARAMS["outputs"]["output_path"], PARAMS["outputs"]["order_id"],PARAMS["outputs"]["optical_path"],
                            PARAMS["outputs"]["sar_path"])
    logging.info("Output Dir Created Successfully")
    ###############################################################################################################
    # SETUP #######################################################################################################
    ###############################################################################################################

    ###############################################################################################################
    # INPUT PREPROCESS ############################################################################################
    ###############################################################################################################
    # Load Bands
    optical_path = PARAMS["inputs"]["optical_path"]
    sar_path = PARAMS["inputs"]["sar_path"]

    try:
        I_1_bands, I_2_bands = LoadBands(optical_path, sar_path, options=options_common)
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
    
            
        
    logging.info("Bands loaded Successfully")  
    # Preprocess Images
    # Crop
    CropStart = PARAMS["preprocessing"]["crop"]["crop_start"]
    CropSize = list(PARAMS["preprocessing"]["crop"]["crop_size"])

    # I_1 = CropBands(I_1_bands, CropStart, CropSize)
    I_1 = CropBands(I_1_bands, CropStart, CropSize, ignore3D=True)
    I_2 = CropBands(I_2_bands, CropStart, CropSize, ignore3D=True)
    I_2copy = I_2
    if options_common["verbose"]:
        print("Optical Image:", I_1.shape)
        print("SAR Image:", I_2.shape)

    # Normalise and Clip
    NormFuncName_1 = PARAMS["preprocessing"]["normalise"]["optical"]["func"]
    NormFuncParams_1 = PARAMS["preprocessing"]["normalise"]["optical"]["params"]
    NormFuncName_2 = PARAMS["preprocessing"]["normalise"]["sar"]["func"]
    NormFuncParams_2 = PARAMS["preprocessing"]["normalise"]["sar"]["params"]
    NormFunc_1 = GetParameterisedNormFunc(NormFuncName_1, NormFuncParams_1)
    NormFunc_2 = GetParameterisedNormFunc(NormFuncName_2, NormFuncParams_2)

    # Optical
    I_1_pan = I_1
    I_1_pan = NormFunc_1(I_1_pan)
    I_1_pan = ClipImageValues(I_1_pan)
    # SAR
    I_2 = NormFunc_2(I_2)
    I_2 = ClipImageValues(I_2)
    I_2_gray = I_2

    # Create Display Images and Display
    # Optical
    I_1_pan_display = np.array(I_1_pan * 255, dtype=np.uint8)
    options = dict(options_optical)
    options["path"] = options_optical["path"].format("Optical_Bands")
    ShowImages_Grid([I_1_pan], 
                    nCols=1, 
                    titles=["Optical (Pan)"], 
                    figsize=(10, 10), gap=(0.25, 0.25), 
                    options=options)
    options["path"] = options_optical["path"].format("Optical_Pan_Histogram")
    PlotImageHistogram(I_1_pan, 1000, "Optical_Histogram", options)
    # SAR
    I_2_display = np.array(I_2 * 255, dtype=np.uint8)
    I_2_gray_display = np.array(I_2_gray * 255, dtype=np.uint8)
    options = dict(options_sar)
    options["path"] = options_sar["path"].format("SAR_Bands")
    ShowImages_Grid([I_2_gray], 
                    nCols=2, 
                    titles=["SAR"], 
                    figsize=(10, 10), gap=(0.25, 0.25), 
                    options=options)
    options["path"] = options_sar["path"].format("SAR_Histogram")
    PlotImageHistogram(I_2_gray, 1000, "SAR_Histogram", options)

    # Calculate Noise Metrics
    if options_common["verbose"]:
        # Optical
        print("SNR 1 (Pan):", SNR(I_1_pan.ravel()))
        print("ENL 1 (Pan):", ENL(I_1_pan.ravel()))
        # SAR
        print("SNR 2 (Grayscale):", SNR(I_2_gray.ravel()))
        print("ENL 2:", ENL(I_2_gray.ravel()))

    # SAR Denoising
    DenoiserFuncName = PARAMS["preprocessing"]["denoise"]["sar"]["func"]
    DenoiserParams = PARAMS["preprocessing"]["denoise"]["sar"]["params"]
    DenoiserFunc = GetParameterisedDenoiserFunc(DenoiserFuncName, DenoiserParams)
    I_2_gray_denoised = DenoiserFunc(I_2_gray)
    I_2_gray_denoised = NORMALISERS["MinMax"](I_2_gray_denoised)
    # SAR Denoised Normalise
    DenoisedNormFuncName_2 = PARAMS["preprocessing"]["denoised_normalise"]["func"]
    DenoisedNormFuncParams_2 = PARAMS["preprocessing"]["denoised_normalise"]["params"]
    if PARAMS["preprocessing"]["denoised_normalise"]["func"] == "Histogram_Matching":
        DenoisedNormFuncParams_2["ref"] = I_1_pan
    DenoisedNormFunc_2 = GetParameterisedNormFunc(DenoisedNormFuncName_2, DenoisedNormFuncParams_2)
    I_2_gray_denoised_norm = DenoisedNormFunc_2(I_2_gray_denoised)
    # Set Final SAR
    I_2 = I_2_gray_denoised_norm
    I_2_gray = I_2_gray_denoised_norm
    I_2_display = np.array(I_2 * 255, dtype=np.uint8)
    I_2_gray_display = np.array(I_2_gray * 255, dtype=np.uint8)

    # Display Denoised and Norm SAR
    # ENL EPI
    if options_common["verbose"]:
        print("ENL Original:", ENL(I_2_gray.ravel()))
        print("ENL Final:", ENL(I_2_gray_denoised_norm.ravel()))
        print("EPI:", EPI(I_2_gray, I_2_gray_denoised_norm))
    options = dict(options_sar)
    options["path"] = options_sar["path"].format("Original_Denoised_Normalised_SAR")
    ShowImages_Grid([I_2_gray, I_2_gray_denoised, I_2_gray_denoised_norm], nCols=3, 
                    titles=["SAR", "Denoised", "Denoised " + DenoisedNormFuncName_2], 
                    figsize=(25, 25), gap=(0.25, 0.25),
                    options=options)
    # Display Final SAR
    options = dict(options_sar)
    options["path"] = options_sar["path"].format("SAR_Denoised_Bands")
    ShowImages_Grid([I_2_gray], 
                    nCols=2, 
                    titles=["SAR_Denoised"], 
                    figsize=(10, 10), gap=(0.25, 0.25), 
                    options=options)
    options["path"] = options_sar["path"].format("SAR_Denoised_Histogram")
    PlotImageHistogram(I_2_gray, 1000, "SAR_Denoised_Histogram", options)

    # Final Input Images
    # Final Side By Side Input Images
    options = dict(options_common)
    options["path"] = options["path"].format("Input_Images")
    ShowImages_Grid([I_1_pan, I_2_gray], 
                    nCols=2, 
                    titles=["Optical", "SAR"], 
                    figsize=(20, 20), gap=(0.25, 0.25), 
                    options=options)
    ###############################################################################################################
    # INPUT PREPROCESS ############################################################################################
    ###############################################################################################################

    ###############################################################################################################
    # OVERALL PARAMS ##############################################################################################
    ###############################################################################################################
    # Scale Space
    ScaleSpaceParams = {
        "sigma": PARAMS["params"]["scale_space"]["sigma"],
        "S": PARAMS["params"]["scale_space"]["S"],
        "ratio": PARAMS["params"]["scale_space"]["ratio"],
        "d": PARAMS["params"]["scale_space"]["d"],
        "GKernelSize": PARAMS["params"]["scale_space"]["GKernelSize"],
        "diffusion": PARAMS["params"]["scale_space"]["diffusion"]
    }
    use_harris = PARAMS["params"]["scale_space"]["use_harris"]

    # Keypoint Detection
    targetKeypoints_1 = PARAMS["params"]["keypoint_detection"]["optical"]["N_keypoints"]
    targetKeypoints_2 = PARAMS["params"]["keypoint_detection"]["sar"]["N_keypoints"]
    targetKeypoints = [targetKeypoints_1, targetKeypoints_2]
    thresholds = [
        [-0.001, 1.0], 
        [-0.001, 1.0]
    ]
    detectParams_optical = {
        "thresholds": {
            "log_threshold": PARAMS["params"]["keypoint_detection"]["optical"]["thresholds"]["log_threshold"]
        },
        "windows": {
            "extrema_window": PARAMS["params"]["keypoint_detection"]["optical"]["windows"]["extrema_window"],
            "common_check_window": PARAMS["params"]["keypoint_detection"]["optical"]["windows"]["common_check_window"],
            "overall_check_window": PARAMS["params"]["keypoint_detection"]["optical"]["windows"]["overall_check_window"]
        },
        "other": {
            "ignores": PARAMS["params"]["keypoint_detection"]["optical"]["other"]["ignores"],
            "combine_weights": PARAMS["params"]["keypoint_detection"]["optical"]["other"]["combine_weights"]
        },
        "min_scales_detection": PARAMS["params"]["keypoint_detection"]["optical"]["min_scales_detection"],
        "combined_method": PARAMS["params"]["keypoint_detection"]["optical"]["combined_method"]
    }
    detectParams_sar = {
        "thresholds": {
            "log_threshold": PARAMS["params"]["keypoint_detection"]["sar"]["thresholds"]["log_threshold"]
        },
        "windows": {
            "extrema_window": PARAMS["params"]["keypoint_detection"]["sar"]["windows"]["extrema_window"],
            "common_check_window": PARAMS["params"]["keypoint_detection"]["sar"]["windows"]["common_check_window"],
            "overall_check_window": PARAMS["params"]["keypoint_detection"]["sar"]["windows"]["overall_check_window"]
        },
        "other": {
            "ignores": PARAMS["params"]["keypoint_detection"]["sar"]["other"]["ignores"],
            "combine_weights": PARAMS["params"]["keypoint_detection"]["sar"]["other"]["combine_weights"]
        },
        "min_scales_detection": PARAMS["params"]["keypoint_detection"]["sar"]["min_scales_detection"],
        "combined_method": PARAMS["params"]["keypoint_detection"]["sar"]["combined_method"]
    }
    detectParams = [detectParams_optical, detectParams_sar]
    recursions = [
        PARAMS["params"]["keypoint_detection"]["optical"]["adaptive_thresholding"]["recursions"],
        PARAMS["params"]["keypoint_detection"]["sar"]["adaptive_thresholding"]["recursions"]
    ]
    N = [
        PARAMS["params"]["keypoint_detection"]["optical"]["adaptive_thresholding"]["N"],
        PARAMS["params"]["keypoint_detection"]["sar"]["adaptive_thresholding"]["N"]
    ]
    repeatabilityDistThreshold = PARAMS["params"]["keypoint_detection"]["repeatabilityDistThreshold"]

    # Descriptor Generation
    DenseParam = "None"
    if PARAMS["params"]["descriptor_generation"]["dense"]["optical"] and PARAMS["params"]["descriptor_generation"]["dense"]["sar"]:
        DenseParam = "Both"
    elif PARAMS["params"]["descriptor_generation"]["dense"]["optical"]:
        DenseParam = "Optical"
    elif PARAMS["params"]["descriptor_generation"]["dense"]["sar"]:
        DenseParam = "SAR"
    DescriptorMethod = PARAMS["params"]["descriptor_generation"]["descriptor"]["func"]
    DescriptorParams = PARAMS["params"]["descriptor_generation"]["descriptor"]["params"]
    norm = PARAMS["params"]["descriptor_generation"]["norm"]
    includeDescriptor = PARAMS["params"]["descriptor_generation"]["includeDescriptor"]
    includePosition = PARAMS["params"]["descriptor_generation"]["includePosition"]
    positionMultiplier = PARAMS["params"]["descriptor_generation"]["positionMultiplier"] 
    # Match
    MatchMethod = PARAMS["params"]["match"]["func"]
    ImageRegisterParams = {
        "DistFunc_CV2": cv2.NORM_L2,
        "crossCheck": PARAMS["params"]["match"]["params"]["crossCheck"],

        "DistFunc": DISTANCE_METRICS[PARAMS["params"]["match"]["params"]["DistFunc"]],
        "search_window": PARAMS["params"]["match"]["params"]["search_window"],

        "MatchType": PARAMS["params"]["match"]["params"]["MatchType"],
        "RANSAC": PARAMS["params"]["match"]["params"]["RANSAC"],
    }

    # Shift Correction
    ShiftMethod = PARAMS["params"]["shift"]["func"]
    ShiftParams = {
        "HistogramShift_TopKDirections": {
            "N_directions": PARAMS["params"]["shift"]["params"]["HistogramShift_TopKDirections"]["N_directions"],
            "K": PARAMS["params"]["shift"]["params"]["HistogramShift_TopKDirections"]["K"],
            "topKHist": PARAMS["params"]["shift"]["params"]["HistogramShift_TopKDirections"]["topKHist"],
            "transformMethod": PARAMS["params"]["shift"]["params"]["HistogramShift_TopKDirections"]["transformMethod"]
        }
    }

    # Final Display
    resolution = PARAMS["params"]["final_display"]["resolution"]

    # Steps
    Scale_Space_Gen = True
    Detect_Feature_Pts = True
    Descriptor_Gen = True
    Match = True
    RunSteps = [Scale_Space_Gen , Detect_Feature_Pts, Descriptor_Gen, Match]
    ###############################################################################################################
    # OVERALL PARAMS ##############################################################################################
    ###############################################################################################################

    ###############################################################################################################
    # OVERALL RUN #################################################################################################
    ###############################################################################################################
    # Run
    START_TIME = time.time()

    # Scale Space
    if options_common["verbose_main"]:
        print("Generating Scale Space...")
    if RunSteps[0]:
        ScaleSpaceData_1 = HarrisLaplace_ConstructScaleSpace(I_1_pan, ScaleSpaceParams, "Optical", options_optical)
        ScaleSpaceData_2 = HarrisLaplace_ConstructScaleSpace(I_2_gray, ScaleSpaceParams, "SAR-Sobel", options_sar)

    if options_common["plot"] or options_common["save"]:
        # Optical
        DisplayGradients(ScaleSpaceData_1, options_optical)
        DisplayGradientHistograms(ScaleSpaceData_1, options_optical)
        # SAR
        DisplayGradients(ScaleSpaceData_2, options_sar)
        DisplayGradientHistograms(ScaleSpaceData_2, options_sar)

    if not use_harris:
        ScaleSpaceData_1["HarrisResponses"] = ScaleSpaceData_1["gradients"]
        ScaleSpaceData_2["HarrisResponses"] = ScaleSpaceData_2["gradients"]

    # Feature Detection
    if options_common["verbose_main"]:
        print("Detecting Keypoints...")
    if RunSteps[1]:
        KeypointsData_1, harris_threshold_1, targetClosest_1 = AdaptiveHarrisThreshold(ScaleSpaceData_1, targetKeypoints[0], 
                detectParams[0], 
                recursions[0], thresholds[0][0], thresholds[0][1], N[0], options_optical)
        KeypointsData_2, harris_threshold_2, targetClosest_2 = AdaptiveHarrisThreshold(ScaleSpaceData_2, targetKeypoints[1], 
                detectParams[1], 
                recursions[1], thresholds[1][0], thresholds[1][1], N[1], options_sar)
 
    # Plot Scale Space Keypoint Detection
    if options_common["plot"] or options_common["save"]:
        # Optical
        I_keypoints_1_overallCommon = DisplayScaleSpaceDetectedKeypoints(I_1_pan_display, KeypointsData_1, ScaleSpaceData_1, 
                                        detectParams[0]["combined_method"], options_optical)
        # SAR
        I_keypoints_2_overallCommon = DisplayScaleSpaceDetectedKeypoints(I_2_display, KeypointsData_2, ScaleSpaceData_2, 
                                        detectParams[1]["combined_method"], options_sar)

    # Plot Final Keypoints
    # Overall Keypoints
    I_keypoints_1_overallCommon, keypoints_1 = PlotThresholdedKeyPoints(KeypointsData_1["Overall_Common_Keypoints"], I_1_pan_display)
    I_keypoints_2_overallCommon, keypoints_2 = PlotThresholdedKeyPoints(KeypointsData_2["Overall_Common_Keypoints"], I_2_display)
    options = dict(options_common)
    options["path"] = options["path"].format("Common_Keypoints")
    ShowImages_Grid([I_keypoints_1_overallCommon, I_keypoints_2_overallCommon], 
                    nCols=2, 
                    titles=["Common Optical " + str(KeypointsData_1["Overall_Common_Keypoints"].shape[0]), 
                        "Common SAR " + str(KeypointsData_2["Overall_Common_Keypoints"].shape[0])], 
                    figsize=(20, 20), gap=(0.25, 0.25), 
                    options=options)
    logging.info("Feature Detection Successfull")  
    # Show Repeatability
    if options_common["verbose"]:
        Repeatability, N_corr = CalculateRepeatability(KeypointsData_1["Overall_Common_Keypoints"], KeypointsData_2["Overall_Common_Keypoints"], threshold=repeatabilityDistThreshold)
        print("N Correlated:", N_corr, "/", KeypointsData_1["Overall_Common_Keypoints"].shape[0]*KeypointsData_2["Overall_Common_Keypoints"].shape[0])
        print("Repeatability:", Repeatability)

    # Descriptor Generation
    if options_common["verbose_main"]:
        print("Generating Descriptor Data...")
    if RunSteps[2]:
        DescriptorData_1 = {
            "I": I_1_pan, 
            "Responses": None, 
            "Angles": None, 
            "search_window": ImageRegisterParams["search_window"]
        }
        DescriptorData_2 = {
            "I": I_2_gray, 
            "Responses": None, 
            "Angles": None, 
            "search_window": ImageRegisterParams["search_window"]
        }

        if options_common["verbose_main"]:
            print("Generating Descriptors...")
        # Dense Descriptors
        # If Dense, the keypoints around which to generate is taken from the other image (Optical takes from SAR and vice versa)
        if DenseParam == "Optical":
            descriptors_1, KeypointsData_1_final = HarrisLaplace_CalculateWindowedDenseDescriptors(KeypointsData_2, DescriptorData_1, DescriptorMethod, DescriptorParams, norm, options_optical)
            descriptors_2 = HarrisLaplace_CalculateDescriptors(KeypointsData_2, DescriptorData_2, DescriptorMethod, DescriptorParams, norm, options_sar)
            KeypointsData_2_final = KeypointsData_2
        elif DenseParam == "SAR":
            descriptors_1 = HarrisLaplace_CalculateDescriptors(KeypointsData_1, DescriptorData_1, DescriptorMethod, DescriptorParams, norm, options_optical)
            descriptors_2, KeypointsData_2_final = HarrisLaplace_CalculateWindowedDenseDescriptors(KeypointsData_1, DescriptorData_2, DescriptorMethod, DescriptorParams, norm, options_sar)
            KeypointsData_1_final = KeypointsData_1
        elif DenseParam == "Both":
            descriptors_1, KeypointsData_1_final = HarrisLaplace_CalculateWindowedDenseDescriptors(KeypointsData_1, DescriptorData_1, DescriptorMethod, DescriptorParams, norm, options_optical)
            descriptors_2, KeypointsData_2_final = HarrisLaplace_CalculateWindowedDenseDescriptors(KeypointsData_2, DescriptorData_2, DescriptorMethod, DescriptorParams, norm, options_sar)
        else:
            descriptors_1 = HarrisLaplace_CalculateDescriptors(KeypointsData_1, DescriptorData_1, DescriptorMethod, DescriptorParams, norm, options_optical)
            descriptors_2 = HarrisLaplace_CalculateDescriptors(KeypointsData_2, DescriptorData_2, DescriptorMethod, DescriptorParams, norm, options_sar)
            KeypointsData_1_final = KeypointsData_1
            KeypointsData_2_final = KeypointsData_2

        positionDescriptors_1 = HarrisLaplace_CalculateDescriptors(KeypointsData_1, DescriptorData_1, "Position", DescriptorParams, False, options_optical)
        positionDescriptors_2 = HarrisLaplace_CalculateDescriptors(KeypointsData_2, DescriptorData_2, "Position", DescriptorParams, False, options_sar)
    logging.info("Feature Description Successfull")  
    # Match
    if options_common["verbose_main"]:
        print("Matching Descriptors...")
    options = dict(options_common)
    if RunSteps[3]:
        match_descriptors_1 = None
        match_descriptors_2 = None
        if includePosition and includeDescriptor:
            match_descriptors_1 = np.hstack((positionDescriptors_1 * positionMultiplier, descriptors_1))
            match_descriptors_2 = np.hstack((positionDescriptors_2 * positionMultiplier, descriptors_2))
        elif includePosition:
            match_descriptors_1 = np.copy(positionDescriptors_1)
            match_descriptors_2 = np.copy(positionDescriptors_2)
        else:
            match_descriptors_1 = np.copy(descriptors_1)
            match_descriptors_2 = np.copy(descriptors_2)

        if options_common["verbose"]:
            print("Final Descriptor 1:", match_descriptors_1.shape)
            print("Final Descriptor 2:", match_descriptors_2.shape)
        if DenseParam == "None":
            PlotDescriptors(match_descriptors_1, figsize=(25, 10), title="Optical Descriptors")
            PlotDescriptors(match_descriptors_2, figsize=(25, 10), title="SAR Descriptors")
            Descriptor_DistanceMatrix = CalculateDistanceMatrix(match_descriptors_1, match_descriptors_2, ImageRegisterParams["DistFunc"])
            PlotDistanceMatrix(Descriptor_DistanceMatrix, figsize=(10, 10), title="Full Descriptor Distance Matrix")

        # Regenerate Keypoints
        locs_1 = KeypointsData_1_final["Overall_Common_Keypoints"]
        locs_2 = KeypointsData_2_final["Overall_Common_Keypoints"]    
        I_keypoints_1_overallCommon, keypoints_1 = PlotThresholdedKeyPoints(locs_1, I_1_pan_display)
        I_keypoints_2_overallCommon, keypoints_2 = PlotThresholdedKeyPoints(locs_2, I_2_display)
        keypointsDraw = [True, True]
        if DenseParam == "Optical":
            I_keypoints_1_overallCommon = np.copy(I_1_pan_display)
            keypointsDraw[0] = False
        elif DenseParam == "SAR":
            I_keypoints_2_overallCommon = np.copy(I_2_gray_display)
            keypointsDraw[1] = False
        elif DenseParam == "Both":
            I_keypoints_1_overallCommon = np.copy(I_1_pan_display)
            I_keypoints_2_overallCommon = np.copy(I_2_gray_display)
            keypointsDraw = [False, False]

        if MatchMethod == "Template":
            I_2 = NORMALISERS["MinMax"](I_2copy)
            I_2 = NORMALISERS["Histogram_Norm"](I_2)
            I_2 = ClipImageValues(I_2)
            
            # Create Display Images and Display
            # Optical
            I_2_display = np.array(I_2 * 255, dtype=np.uint8)
            r = rasterio.open(optical_path).read(1) #r
            r = CropBands(r, CropStart, CropSize, ignore3D=True)
            g = rasterio.open(optical_path).read(2)
            g = CropBands(g, CropStart, CropSize, ignore3D=True)
            b = rasterio.open(optical_path).read(3)
            b = CropBands(b, CropStart, CropSize, ignore3D=True)
            rgb_list = [r, g, b]
            rgb_norm_list = []
            for band in rgb_list:
                band_norm = NORMALISERS["MinMax"](band)
                band_norm = NORMALISERS["Histogram_Norm"](band_norm)
                rgb_norm_list.append(band_norm * 255)

            I_rgb_norm = np.dstack(tuple(rgb_norm_list))
            sar_norm_list = [I_2_display,I_2_display,I_2_display]
            I_sar_norm = np.dstack(tuple(sar_norm_list))
            I_matched, matchIndices, matchDistances = ImageRegister_TemplateMatch(keypoints_1, keypoints_2, match_descriptors_1, match_descriptors_2, np.uint8(I_rgb_norm), np.uint8(I_sar_norm), ImageRegisterParams, options)

            
            # Generate Template Visualisations
            options_temp = dict(options)
            I_1_searchWindow = GetSearchWindowedImage(I_keypoints_1_overallCommon, locs_1, 
                                    search_window=ImageRegisterParams["search_window"])
            I_2_searchWindow = GetSearchWindowedImage(I_keypoints_2_overallCommon, locs_1, 
                                    search_window=ImageRegisterParams["search_window"])
            
            I_match_comparison = GetKeypointComparisonImage(I_1_pan_display, locs_1,
                keypoints_1, keypoints_2, matchIndices, ImageRegisterParams["search_window"], 
                                    keypointsDraw=keypointsDraw)
            options_temp["path"] = options["path"].format("TemplateWindows_Comparison")
            ShowImages_Grid([I_1_searchWindow, I_2_searchWindow], 
                            nCols=2, 
                            titles=["Template Windows Optical", "Search Windows SAR"], 
                            figsize=(20, 20), gap=(0.25, 0.25), 
                            options=options_temp)
            options_temp["path"] = options["path"].format("Match_Comparison")
            ShowImages_Grid([I_match_comparison], 
                            nCols=1, 
                            titles=["Match Comparison"], 
                            figsize=(20, 20), gap=(0.25, 0.25), 
                            options=options_temp)
            options_temp["path"] = options["path"].format("Match_Comparison_SAR_Flip")
            ShowImages_Grid([I_2_gray_display], 
                            nCols=1, 
                            titles=["SAR"], 
                            figsize=(20, 20), gap=(0.25, 0.25), 
                            options=options_temp)
        else:
            I_matched, matchIndices, matchDistances = ImageRegister_CV2(keypoints_1, keypoints_2, match_descriptors_1, match_descriptors_2, 
                                        I_1_pan_display, I_2_display, ImageRegisterParams, options)
    logging.info("Feature Matching Successfull")  
    # Plot and Save Match Image
    options["path"] = options["path"].format("Match")
    ShowImages_Grid([I_matched], 
                    nCols=1, 
                    titles=["Match"], 
                    figsize=(20, 20), gap=(0.25, 0.25), 
                    options=options_temp)
    plt.imsave("./preview.jpg", I_matched, dpi=10000, cmap='gray')
    # Calculate Shift
    if options_common["verbose_main"]:
        print("Calculating Shift...")
    if options_common["verbose"]:
        print("Matches:", matchIndices.shape)
        print("Match Distances:", np.array(matchDistances).shape)
    ShiftParams.update({
        "matchDistances": matchDistances
    })
    OverallTransform = CalculateShift(ShiftMethod, ShiftParams, KeypointsData_1_final["Overall_Common_Keypoints"], KeypointsData_2_final["Overall_Common_Keypoints"], 
                                matchIndices, options=options)
    # Shift SAR Image
    if options_common["verbose"]:
        print("Predicted Transform:", OverallTransform)
    #I_2_gray =  rasterio.open(sar_path).read(1)
    #I_2_gray = CropBands(I_2_gray, CropStart, CropSize, ignore3D=True)
    I_2_gray_corrected = TransformImage(I_2_gray, OverallTransform)
    #I_2_gray_corrected = TransformImage(I_2_gray, OverallTransform)
    #I_2_gray = np.clip(I_2_gray, 20, 500, out=None)
    #I_2_gray_corrected = np.clip(I_2_gray_corrected, 20, 500, out=None)
    # Display Shifted Images
    logging.info("Image Transform Successfull")  
    options["path"] = options_common["path"].format("Correction")
    ShowImages_Grid([I_2_gray, I_2_gray_corrected, I_1_pan], 
                    nCols=3, 
                    titles=["SAR", "SAR Corrected", "Optical"], 
                    figsize=(20, 20), gap=(0.25, 0.25), 
                    options=options)
    # Display Checkerboard Images
    #DisplayCheckerboardComparison(I_1_pan, I_2_gray, I_2_gray_corrected, resolution, options=options_common)

    # Final Outputs
    I_1_bands_norm = NORMALISERS["MinMax"](I_1_bands)
    I_2_bands_norm = NORMALISERS["Histogram_Norm"](I_2_bands)
    #I_2_bands_norm = NORMALISERS["Histogram_Matching"](I_2_bands, ref=I_1_bands_norm)
    I_2_bands_norm_corrected = TransformImage(I_2_bands_norm, OverallTransform)
    options = dict(options_common)
    #options["path"] = options_common["path"].format("Optical_Final")
    #plt.imsave(options["path"], I_1_bands_norm, dpi=10000, cmap='gray')
    # cv2.imwrite(options["path"], I_1_bands_norm)
    options["path"] = options_common["path"]
    print('output_path', options["path"])
    plt.imsave(PARAMS["outputs"]["sar_path"] + 'Sar.jpg', I_2_bands_norm_corrected, dpi=10000, cmap='gray')
    # cv2.imwrite(options["path"], I_2_bands_norm_corrected)
    r = rasterio.open(optical_path).read(1) #r
    g = rasterio.open(optical_path).read(2)
    b = rasterio.open(optical_path).read(3)
    rgb_list = [r, g, b]
    rgb_norm_list = []
    for band in rgb_list:
        band_norm = NORMALISERS["MinMax"](band)
        band_norm = NORMALISERS["Histogram_Norm"](band_norm)
        rgb_norm_list.append(band_norm)

    I_rgb_norm = np.dstack(tuple(rgb_norm_list))
    options["path"] = options_common["path"]
    plt.imsave(PARAMS["outputs"]["optical_path"] + 'Optical.jpg', I_rgb_norm, dpi=10000)
    
    #Store and zip
    I_1_bands, I_2_bands = rasterio.open(PARAMS["inputs"]["noptical_path"]).read(),rasterio.open(sar_path).read()
    op_meta = rasterio.open(PARAMS["inputs"]["noptical_path"]).meta
    sar_meta = rasterio.open(sar_path).meta
    #with rasterio.open(PARAMS["outputs"]["order_id"]+'Optical.tif', 'w', **op_meta) as dst:
     #   dst.write(I_1_bands)
    #print(I_1_bands.shape, I_2_bands.shape)
    
    with rasterio.open(
    PARAMS["outputs"]["optical_path"]+'Optical.tif',
    'w',
    driver='GTiff',
    height=I_1_bands.shape[1],
    width=I_1_bands.shape[2],
    count=op_meta['count'],
    dtype=I_1_bands.dtype) as dst:
        dst.write(I_1_bands)
    
    with rasterio.open(
    PARAMS["outputs"]["sar_path"]+'Sar.tif',
    'w',
    driver='GTiff',
    height=I_2_bands.shape[1],
    width=I_2_bands.shape[2],
    count=sar_meta['count'],
    dtype=I_2_bands.dtype) as dst:
        dst.write(I_2_bands)
    
    
    
   # with rasterio.open(PARAMS["outputs"]["order_id"]+'Sar.tif', 'w', **sar_meta) as dst:
    #    dst.write(I_2_bands.shape)
    if os.environ.get('ORDER_ID') is not None:
        order_id = os.environ.get('ORDER_ID')
    else:
        order_id = "test"
    shutil.make_archive(order_id , 'zip', './Coreg')
    
  

    #s3_client.put_object(
    #    Bucket=PARAMS["outputs"]["s3_bucket"],
    #    Body='',
    #    Key=PARAMS["outputs"]["s3_path"] + order_id +'/')
 

        #s3_client.put_object(Bucket=PARAMS["outputs"]["s3_bucket"], Key=(PARAMS["outputs"]["s3_path"] + order_id +'/'))
        
    s3_client.upload_file("./" + order_id +".zip" , PARAMS["outputs"]["s3_bucket"], PARAMS["outputs"]["s3_path"] + order_id +'/' + order_id + '.zip')
    
    s3_client.upload_file("./preview.jpg" , PARAMS["outputs"]["s3_bucket"], PARAMS["outputs"]["s3_path"] + order_id +'/' + 'preview.jpg')
    s3_client.upload_file(PARAMS["outputs"]["optical_path"] +"Optical.jpg" , PARAMS["outputs"]["s3_bucket"], PARAMS["outputs"]["s3_path"] + order_id +'/' + 'Optical.jpg')
    s3_client.upload_file(PARAMS["outputs"]["sar_path"] +'Sar.jpg' , PARAMS["outputs"]["s3_bucket"], PARAMS["outputs"]["s3_path"] + order_id +'/' + 'Sar.jpg')
        #s3_client.put_object(Bucket=PARAMS["outputs"]["s3_bucket"], Key=(PARAMS["outputs"]["s3_path"] + PARAMS["outputs"]["output_path"][2:] ))
        
        #s3_client.upload_file( options_common["path"] + 'Optical.jpg', PARAMS["outputs"]["s3_bucket"], PARAMS["outputs"]["s3_path"] + PARAMS["outputs"]["order_id"]+'/' + "Optical.jpg")
        #s3_client.upload_file(PARAMS["outputs"]["output_path"] + 'Sar.jpg', PARAMS["outputs"]["s3_bucket"], PARAMS["outputs"]["s3_path"] +  PARAMS["outputs"]["order_id"]+'/' + "Sar.jpg")

    if os.environ.get('ORDER_ID') is not None:
        order_id = os.environ.get('ORDER_ID')
        email_id = os.environ.get('EMAIL_ID')
    else:
        order_id = "test"
    client = boto3.client('ses')
    client.send_email(
    Source= 'support@galaxeye.space',
    Destination= {
      'ToAddresses': [f'{email_id}'],
    },
     Message={
        'Subject': {
        'Data': 'Your order has been processed',
        'Charset': 'UTF-8',
      },
        'Body': {
            'Html': {
                'Charset': 'UTF-8',
                'Data': f'Your data with {order_id} has been processed and is ready for download. Please click on the link below to download the data. <a href="http://localhost:4000/order/{order_id}">Download</a>'
                }
            }
        },
    )
    if os.environ.get('ORDER_ID') is not None:
        order_id = os.environ.get('ORDER_ID')
    else:
        order_id = "test"
    query_handler(order_id, "Processed")

    END_TIME = time.time()
# if options_common["verbose_main"]:
#         print("TOTAL TIME:", FormatTime(END_TIME - START_TIME))
    ###############################################################################################################
    # OVERALL RUN #################################################################################################
    ###############################################################################################################    

# Driver Code
if __name__ == "__main__":
    ###############################################################################################################
    # LOADING #####################################################################################################
    ###############################################################################################################
    # Load Params
    import params
    PARAMS = params.PARAMS

    # Load Sys Params
    PARAMS["outputs"]["s3"] = False
    if len(sys.argv) >= 2:
        PARAMS["inputs"]["optical_path"] = sys.argv[1]
    if len(sys.argv) >= 3:
        PARAMS["inputs"]["sar_path"] = sys.argv[2]
    if len(sys.argv) >= 4:
        PARAMS["outputs"]["output_path"] = sys.argv[3]
    ###############################################################################################################
    # LOADING #####################################################################################################
    ###############################################################################################################

    # Run Coreg V2
    RunCoRegV2(PARAMS)