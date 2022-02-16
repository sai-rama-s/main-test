"""
Keypoints Detection
"""

# Imports
import numpy as np
import cv2
import time

from .utils import *

## Utils Functions
def CalculateRepeatability(locs_1, locs_2, threshold=1.5):
    '''
    Calculates the repeatability of two sets of keypoints.
    '''

    n1 = locs_1.shape[0]
    n2 = locs_2.shape[0]
    N_corr = 0
    for loc_1 in tqdm(locs_1):
        for loc_2 in locs_2:
            dist = np.sqrt(np.sum((loc_1 - loc_2)**2))
            if dist < threshold:
                N_corr += 1
    repeatability = (2 * N_corr) / (n1 + n2)
    return repeatability, N_corr

# Extrema Detection Functions
def FindLocalMaximas(Response, threshold, window=1, BaseKeypoints=None):
    '''
    Finds local maximas in the response map.
    '''

    M, N = Response.shape
    BORDER_WIDTH = window

    # Scale
    Response = NORMALISERS["MinMax"](Response)

    # Find
    Keypoints = []

    # If no Base Keypoints
    if BaseKeypoints is None:
        for j in range(BORDER_WIDTH, M-BORDER_WIDTH, 1):
            for k in range(BORDER_WIDTH, N-BORDER_WIDTH, 1):
                temp = Response[j, k]
                respWindow = Response[j-window:j+window, k-window:k+window]
                if temp > threshold and np.count_nonzero(respWindow > temp) == 0:
                    KP = [j, k]
                    Keypoints.append(KP)
    # If Base Keypoints are given
    else:
        for kp in BaseKeypoints:
            if Response[kp[0], kp[1]] > threshold:
                Keypoints.append([kp[0], kp[1]])
    
    return Keypoints

def KeypointSelect_Intersection_Thresholded(K1, K2, size, common_check_window=0):
    '''
    Selects keypoints from K1 that are also nearby to keypoints in K2.
    '''

    KMap = np.zeros(size, dtype=int)
    KC = []
    for kp in K1:
        KMap[kp[0], kp[1]] = 1
    for kp in K2:
        if common_check_window > 0:
            if 1 in KMap[kp[0]-common_check_window:kp[0]+common_check_window, kp[1]-common_check_window:kp[1]+common_check_window]:
                KC.append([kp[0], kp[1]])
        elif KMap[kp[0], kp[1]] == 1:
            KC.append([kp[0], kp[1]])
    KC = np.array(KC)
    return KC

def KeypointSelect_Union_Thresholded(CommonKeypoints, overall_check_window=0, min_scales_detection=1):
    '''
    Selects keypoints which appear (nearby) in a specified number of consecutive scales.
    '''

    # Initially do direct Union to get all candidate locations
    CandidateKeypointsMap = np.any(CommonKeypoints, axis=0)
    CandidateKeypoints = Map2Points(CandidateKeypointsMap)

    # For every keypoint check if it has atleast 1 nearby point in atleast min_scales_detection consecutive scales
    FinalKeypoints = []
    FinalKeypointsMap = np.zeros(CommonKeypoints.shape[1:], dtype=bool)
    FoundScales = []
    for kp in CandidateKeypoints:
        # Check if a nearby point has already been added - If so, goto next keypoint
        if overall_check_window > 0:
            if True in FinalKeypointsMap[kp[0]-overall_check_window:kp[0]+overall_check_window, kp[1]-overall_check_window:kp[1]+overall_check_window]:
                continue
        elif FinalKeypointsMap[kp[0], kp[1]] == True:
            continue

        curFoundScalesCount = 0
        curFoundScale = -1
        for i in range(CommonKeypoints.shape[0]):
            if True in CommonKeypoints[i][kp[0]-overall_check_window:kp[0]+overall_check_window, kp[1]-overall_check_window:kp[1]+overall_check_window]:
                    curFoundScalesCount += 1
                    if curFoundScale == -1:
                        curFoundScale = i
            else:
                # If not found, reset curFoundScaleCount (since only consecutive scales detected allowed)
                curFoundScalesCount = 0
                curFoundScale = -1
            # Check if more than min needed
            if curFoundScalesCount >= min_scales_detection:
                FinalKeypoints.append([kp[0], kp[1]])
                FinalKeypointsMap[kp[0], kp[1]] = True
                FoundScales.append(curFoundScale)
                break
        
    return FinalKeypoints, FoundScales

# Threshold Functions
def ThresholdKeypoints(ScaleSpaceData, thresholds, windows, min_scales_detection=1, ignores=(False, False), BaseKeypoints={}):
    ''' 
    Thresholds the keypoints in the scale space.
    thresholds[harris_threshold] : Threshold for Harris Responses
    thresholds[log_threshold] : Threshold for LoG Responses
    windows[extrema_window] : Extrema Window to check for maxima keypoint
    windows[common_check_window] : Min Dist for selection of nearby point between Harris and LoG
    windows[overall_check_window] : Min Dist for selection of nearby point between different scales
    min_scales_detection : Min number of scales a point must be detected in to be taken for final keypoints
    BaseKeypoints : Already detected keypoints with 0.0 threshold (used as a base to apply thresholding on top)
    '''

    HarrisResponses = ScaleSpaceData["HarrisResponses"]
    LoGs = ScaleSpaceData["LoGs"]
    harris_threshold = thresholds["harris_threshold"]
    log_threshold = thresholds["log_threshold"]
    extrema_window = windows["extrema_window"]
    common_check_window = windows["common_check_window"]
    overall_check_window = windows["overall_check_window"]

    FeaturesData = {
        "Scalewise_Harris_Keypoints": [],
        "Scalewise_LoG_Keypoints": [],
        "Scalewise_Common_Keypoints": [],
        "Overall_Harris_Keypoints": [],
        "Overall_LoG_Keypoints": [],
        "Overall_Common_Keypoints": [],
        "Overall_Common_Keypoints_Scales": []
    }

    HarrisKeypointsMap = np.zeros(LoGs.shape, dtype=bool)
    LoGKeypointsMap = np.zeros(LoGs.shape, dtype=bool)
    CommonKeypointsMap = np.zeros(LoGs.shape, dtype=bool)
    for i in range(HarrisResponses.shape[0]):
        # Threshold Harris Response
        if "Harris" not in BaseKeypoints.keys():
            Keypoints_Harris = FindLocalMaximas(HarrisResponses[i], harris_threshold, extrema_window)
        else:
            Keypoints_Harris = FindLocalMaximas(HarrisResponses[i], harris_threshold, extrema_window, BaseKeypoints["Harris"][i])
        # Threshold LoGs
        if "LoG" not in BaseKeypoints.keys():
            Keypoints_LoGs = FindLocalMaximas(LoGs[i], log_threshold, extrema_window)
        else:
            Keypoints_LoGs = FindLocalMaximas(LoGs[i], log_threshold, extrema_window, BaseKeypoints["LoG"][i])

        # Common Keypoints
        if ignores[1]:
            Keypoints_Common = Keypoints_Harris
        elif ignores[0]:
            Keypoints_Common = Keypoints_LoGs
        else:
            Keypoints_Common = KeypointSelect_Intersection_Thresholded(Keypoints_Harris, Keypoints_LoGs, LoGs.shape[1:], common_check_window)      

        FeaturesData["Scalewise_Harris_Keypoints"].append(Keypoints_Harris)
        FeaturesData["Scalewise_LoG_Keypoints"].append(Keypoints_LoGs)
        FeaturesData["Scalewise_Common_Keypoints"].append(Keypoints_Common)

        for kp in Keypoints_Harris:
            HarrisKeypointsMap[i][kp[0], kp[1]] = True
        for kp in Keypoints_LoGs:
            LoGKeypointsMap[i][kp[0], kp[1]] = True
        for kp in Keypoints_Common:
            CommonKeypointsMap[i][kp[0], kp[1]] = True

    FinalHarrisKeypoints, _ = KeypointSelect_Union_Thresholded(HarrisKeypointsMap, overall_check_window, min_scales_detection)
    FinalLoGKeypoints, _ = KeypointSelect_Union_Thresholded(LoGKeypointsMap, overall_check_window, min_scales_detection)
    FinalKeypoints, FoundScales = KeypointSelect_Union_Thresholded(CommonKeypointsMap, overall_check_window, min_scales_detection)

    FeaturesData["Overall_Harris_Keypoints"] = FinalHarrisKeypoints
    FeaturesData["Overall_LoG_Keypoints"] = FinalLoGKeypoints
    FeaturesData["Overall_Common_Keypoints"] = FinalKeypoints
    FeaturesData["Overall_Common_Keypoints_Scales"] = FoundScales

    for k in FeaturesData.keys():
        FeaturesData[k]= np.array(FeaturesData[k])
    
    return FeaturesData

def ThresholdKeypoints_CombinedMeasure(ScaleSpaceData, thresholds, windows, min_scales_detection=1, combine_weights=(0.5, 0.5), BaseKeypoints={}):
    '''
    Thresholds the keypoints in the scale space using a combined measure.
    thresholds[combined_threshold] : Threshold for Combined Response
    combine_weights : Weights for combining Harris and LoG
    windows[extrema_window] : Extrema Window to check for maxima keypoint
    windows[overall_check_window] : Min Dist for selection of nearby point between different scales
    min_scales_detection : Min number of scales a point must be detected in to be taken for final keypoints
    '''

    HarrisResponses = ScaleSpaceData["HarrisResponses"]
    LoGs = ScaleSpaceData["LoGs"]
    combined_threshold = thresholds["combined_threshold"]
    extrema_window = windows["extrema_window"]
    overall_check_window = windows["overall_check_window"]

    

    FeaturesData = {
        "Scalewise_CombinedResponse": [],
        "Scalewise_Common_Keypoints": [],
        "Overall_Common_Keypoints": [],
        "Overall_Common_Keypoints_Scales": []
    }

    CombinedKeypointsMap = np.zeros(LoGs.shape, dtype=bool)
    for i in range(HarrisResponses.shape[0]):
        # Threshold Combined Response
        CombinedResponse = combine_weights[0] * HarrisResponses[i] + combine_weights[1] * LoGs[i]

        if "Combined" not in BaseKeypoints.keys():
            Keypoints_Combined = FindLocalMaximas(CombinedResponse, combined_threshold)
        else:
            Keypoints_Combined = FindLocalMaximas(CombinedResponse, combined_threshold, BaseKeypoints["Combined"][i])

        FeaturesData["Scalewise_CombinedResponse"].append(np.copy(CombinedResponse))
        FeaturesData["Scalewise_Common_Keypoints"].append(Keypoints_Combined)

        for kp in Keypoints_Combined:
            CombinedKeypointsMap[i][kp[0], kp[1]] = True

    FinalKeypoints, FoundScales = KeypointSelect_Union_Thresholded(CombinedKeypointsMap, overall_check_window, min_scales_detection)

    FeaturesData["Overall_Common_Keypoints"] = FinalKeypoints
    FeaturesData["Overall_Common_Keypoints_Scales"] = FoundScales

    for k in FeaturesData.keys():
        FeaturesData[k]= np.array(FeaturesData[k])
    
    return FeaturesData

# Display Functions
def PlotThresholdedKeyPoints(locs, I):
    '''
    Plots the keypoints on the image
    '''

    locs = np.array(locs)
    if locs.shape[0] == 0:
        return I, []
    locs = np.dstack((locs[:, 1], locs[:, 0]))[0]
    keypoints = GetCv2KeyPoints(locs, posOnly=True)
    I_keypoints = cv2.drawKeypoints(I, keypoints, 0, (0, 255, 0))
    
    return I_keypoints, keypoints

def DisplayScaleSpaceDetectedKeypoints(I_display, FeaturesData, ScaleSpaceData=None, combinedMeasure=False, options=DEFAULT_OPTIONS):
    '''
    Displays the detected keypoints across the scales
    '''
    options = dict(options)
    pathFormat = options["path"]
    
    # Plot Overall Keypoints
    options["path"] = pathFormat.format("KeypointDetection_Overall_Keypoints")
    if not combinedMeasure:
        I_keypoints_overallCommon, keypoints = PlotThresholdedKeyPoints(FeaturesData["Overall_Common_Keypoints"], I_display)
        I_keypoints_overallHarris, _ = PlotThresholdedKeyPoints(FeaturesData["Overall_Harris_Keypoints"], I_display)
        I_keypoints_overallLoG, _ = PlotThresholdedKeyPoints(FeaturesData["Overall_LoG_Keypoints"], I_display)
        ShowImages_Grid([I_keypoints_overallCommon, I_keypoints_overallHarris, I_keypoints_overallLoG], 
                            nCols=4, 
                            titles=["Common Overall", "Harris Overall", "LoG Overall"], 
                            figsize=(20, 20), gap=(0.25, 0.25), 
                            options=options)
    else:
        I_keypoints_overallCommon, keypoints = PlotThresholdedKeyPoints(FeaturesData["Overall_Common_Keypoints"], I_display)
        ShowImages_Grid([I_keypoints_overallCommon], 
                            nCols=1, 
                            titles=["Combined Overall"], 
                            figsize=(10, 10), gap=(0.25, 0.25), 
                            options=options)

    # Plot Scale Space Keypoints
    for i in range(FeaturesData["Scalewise_Common_Keypoints"].shape[0]):
        J = I_display
        if ScaleSpaceData is not None:
            J = np.array(NORMALISERS["MinMax"](ScaleSpaceData["Js"][i]) * 255, dtype=np.uint8)

        I_keypoints_scaleCommon, _ = PlotThresholdedKeyPoints(FeaturesData["Scalewise_Common_Keypoints"][i], J)

        options["path"] = pathFormat.format("KeypointDetection_Scalewise_Keypoints_" + str(i+1))
        if not combinedMeasure:
            I_keypoints_scaleHarris, _ = PlotThresholdedKeyPoints(FeaturesData["Scalewise_Harris_Keypoints"][i], J)
            I_keypoints_scaleLoG, _ = PlotThresholdedKeyPoints(FeaturesData["Scalewise_LoG_Keypoints"][i], J)
            ShowImages_Grid([I_keypoints_scaleCommon, I_keypoints_scaleHarris, I_keypoints_scaleLoG], 
                            nCols=4, 
                            titles=["Common " + str(i+1), "Harris " + str(i+1), "LoG " + str(i+1)], 
                            figsize=(20, 20), gap=(0.25, 0.25), 
                            options=options)
        else:
            CombinedResponse = NORMALISERS["MinMax"](FeaturesData["Scalewise_CombinedResponse"][i])
            ShowImages_Grid([I_keypoints_scaleCommon, CombinedResponse], 
                            nCols=2, 
                            titles=["Combined " + str(i+1), "Combined Response"], 
                            figsize=(10, 10), gap=(0.25, 0.25), 
                            options=options)

    return I_keypoints_overallCommon

# Runner Functions
def HarrisLaplace_KeypointDetect(ScaleSpaceData, KeypointDetectParams, options=DEFAULT_OPTIONS):
    '''
    Detects the keypoints across the scale space
    '''
    options = dict(options)
                
    start_time_fd = time.time()

    min_scales_detection = KeypointDetectParams["min_scales_detection"]
    combined_method = KeypointDetectParams["combined_method"]
    thresholds = KeypointDetectParams["thresholds"]
    windows = KeypointDetectParams["windows"]
    otherParams = KeypointDetectParams["other"]
    if not combined_method:
        FeaturesData = ThresholdKeypoints(ScaleSpaceData, thresholds, windows, min_scales_detection, otherParams["ignores"], otherParams["base_keypoints"])
    else:
        FeaturesData = ThresholdKeypoints_CombinedMeasure(ScaleSpaceData, thresholds, windows, min_scales_detection, otherParams["combine_weights"], otherParams["base_keypoints"])

    end_time_fd = time.time()

    if options["verbose"]:
        print("Time for Feature Point Detection:", FormatTime(end_time_fd - start_time_fd, 4))
        print("Number of feature points:", FeaturesData["Overall_Common_Keypoints"].shape[0])

    return FeaturesData

# Adaptive Threshold Functions
def AdaptiveHarrisThreshold(ScaleSpaceData, target, KeypointDetectParams, 
                            recursions=0, start=-0.01, end=1.0, N=10, options=DEFAULT_OPTIONS):
    '''
    Adaptively thresholds the scale space to get target number of keypoints
    '''
    combined_method = KeypointDetectParams["combined_method"]

    # Check if need to generate base_keypoints
    if "base_keypoints" not in KeypointDetectParams["other"].keys():
        if options["verbose_main"]:
            print("Generating Base Keypoints for Threshold:", start)

        KeypointDetectParams["other"]["base_keypoints"] = {}
        if combined_method:
            KeypointDetectParams["thresholds"]["combined_threshold"] = start
        else:
            KeypointDetectParams["thresholds"]["harris_threshold"] = start
        FeaturesData = HarrisLaplace_KeypointDetect(ScaleSpaceData, KeypointDetectParams, options)
        BaseKeypoints = {}
        if combined_method:
            KeypointDetectParams["other"]["base_keypoints"]["Combined"] = FeaturesData["Scalewise_Common_Keypoints"]
        else:
            KeypointDetectParams["other"]["base_keypoints"]["Harris"] = FeaturesData["Scalewise_Harris_Keypoints"]
            KeypointDetectParams["other"]["base_keypoints"]["LoG"] = FeaturesData["Scalewise_LoG_Keypoints"]

    # Continue checking for thresholds
    if options["verbose_main"]:
        print()
        print("Searching:", start, "->", end)
    thresholds = np.linspace(start, end, N)

    N_KP_prev = None
    FeaturesData_prev = None
    for i in tqdm(range(thresholds.shape[0]), disable=not options["verbose"]):
        threshold = thresholds[i]

        if options["verbose"]:
            print("Threshold:", threshold)

        if combined_method:
            KeypointDetectParams["thresholds"]["combined_threshold"] = threshold
        else:
            KeypointDetectParams["thresholds"]["harris_threshold"] = threshold
        FeaturesData = HarrisLaplace_KeypointDetect(ScaleSpaceData, KeypointDetectParams, options)
        N_KP = FeaturesData["Overall_Common_Keypoints"].shape[0]

        if N_KP == target:
            return FeaturesData, threshold, N_KP
        elif N_KP < target:
            if i == 0:
                if recursions > 0:
                    closestFeaturesData, closestThresh, N_KP_closest = AdaptiveHarrisThreshold(ScaleSpaceData, target, 
                            KeypointDetectParams, 
                            recursions-1, start=start, end=threshold, N=N, options=options)
                    if abs(target - N_KP_closest) <= abs(target - N_KP):
                        return closestFeaturesData, closestThresh, N_KP_closest
                    else:
                        return FeaturesData, threshold, N_KP
                else:
                    return FeaturesData, threshold, N_KP
            else:
                if recursions > 0:
                    closestFeaturesData, closestThresh, N_KP_closest = AdaptiveHarrisThreshold(ScaleSpaceData, target, 
                                KeypointDetectParams, 
                                recursions-1, start=thresholds[i-1], end=thresholds[i], N=N, options=options)
                    Counts = np.array([N_KP, N_KP_prev, N_KP_closest])
                    Threshs = np.array([thresholds[i], thresholds[i-1], closestThresh])
                    Datas = np.array([FeaturesData, FeaturesData_prev, closestFeaturesData])
                    minInd = np.argmin(np.abs(target - Counts))
                    return Datas[minInd], Threshs[minInd], Counts[minInd]
                else:
                    if abs(target - N_KP_prev) <= abs(target - N_KP):
                        return FeaturesData_prev, thresholds[i-1], N_KP_prev
                    else:
                        return FeaturesData, threshold, N_KP
        N_KP_prev = N_KP
        FeaturesData_prev = FeaturesData
    
    return None, end, None

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Keypoint Detection!")