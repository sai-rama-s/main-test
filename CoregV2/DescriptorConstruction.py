"""
Descriptor Construction
"""

# Imports
import time
import numpy as np

from .utils import *

# Main Functions
# Window Functions
def GetSearchWindowMask(locs, size, search_window=15):
    '''
    Return a mask where all pixels within search_window of a keypoint are marked True and rest are False
    '''
    mask = np.zeros(size, dtype=bool)
    for i in range(locs.shape[0]):
        x, y = locs[i, 0], locs[i, 1]
        bounds = np.array([
                  [max(0, x-search_window), min(size[0], x+search_window)], 
                  [max(0, y-search_window), min(size[1], y+search_window)]
        ])
        mask[bounds[0, 0]:bounds[0, 1], bounds[1, 0]:bounds[1, 1]] = True
    return mask

# Runner Functions
def HarrisLaplace_CalculateDescriptors(KeypointsData, DescriptorData, DescriptorMethod, DescriptorParams, norm=True, options=DEFAULT_OPTIONS):
    '''
    Calculates the descriptors for the given keypoints using the specified method
    '''
    options = dict(options)
    DescriptorParams["options"] = options
    
    start_time_cd = time.time()

    descriptors = DESCRIPTORS[DescriptorMethod](KeypointsData, DescriptorData, DescriptorParams, norm)

    end_time_cd = time.time()

    if options["verbose"]:
        print("Time for Calculating " + DescriptorMethod + " Descriptors:", FormatTime(end_time_cd - start_time_cd, 4))
        print(DescriptorMethod, "Descriptors:", descriptors.shape)

    return descriptors

def HarrisLaplace_CalculateWindowedDenseDescriptors(KeypointsData, DescriptorData, DescriptorMethod, DescriptorParams, norm=True, options=DEFAULT_OPTIONS):
    '''
    Calculates the dense descriptors for all pixels within search_window of keypoints using the specified method
    '''
    options = dict(options)
    DescriptorParams["options"] = options
    
    start_time_cd = time.time()

    # Get Window Mask and required locations
    denseMask = GetSearchWindowMask(KeypointsData["Overall_Common_Keypoints"], DescriptorData["I"].shape[:2], search_window=DescriptorData["search_window"])
    requiredPoints = Map2Points(denseMask)

    # Construct Modified Keypoints Data
    KeypointsData = {}
    KeypointsData["Overall_Common_Keypoints"] = requiredPoints
    KeypointsData["Overall_Common_Keypoints_Scales"] = np.ones(requiredPoints.shape[0])

    # Get Descriptors
    descriptors = DESCRIPTORS[DescriptorMethod](KeypointsData, DescriptorData, DescriptorParams, norm)

    end_time_cd = time.time()

    if options["verbose"]:
        print("Time for Calculating " + DescriptorMethod + " Descriptors:", FormatTime(end_time_cd - start_time_cd, 4))
        print(DescriptorMethod, "Descriptors:", descriptors.shape)
    if options["plot"]:
        options["path"] = options["path"].format("Dense_Descriptor_Mask")
        ShowImages_Grid([denseMask], 
            nCols=1, 
            titles=["Dense Descriptor Mask"], 
            figsize=(10, 10), gap=(0.25, 0.25),
            options=options)

    return descriptors, KeypointsData

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Descriptor Construction!")