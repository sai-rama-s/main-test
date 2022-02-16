'''
CFOG Descriptor Generation
'''

# Imports
import cv2
import numpy as np

from ..ImageUtils import *
from ..Normalisers import *

# Main Functions
def GetCFOGDescriptors(I, N_ORIENTATIONS=8, sigma=2.0, ksize=3, plot=False, verbose_main=True):
    '''
    Get CFOG Descriptors for every pixel in image
    '''

    Angles = np.linspace(0, np.pi, N_ORIENTATIONS)

    # Calculate Gx Gy
    Gx = cv2.Sobel(I, 6, 1, 0, ksize=3)
    Gy = cv2.Sobel(I, 6, 0, 1, ksize=3)

    # Plot
    if plot:
        ShowImages_Grid([I, Gx, Gy], 
            nCols=3, 
            titles=["I", "Gx", "Gy"], 
            figsize=(10, 10), gap=(0.25, 0.25))

    # Calculate G_Thetas and Gaussian Blurred G_Thetas
    G_thetas = []
    G_thetas_blurred = []
    for angle in tqdm(Angles, disable=not verbose_main):
        G_theta = np.abs(Gx*np.cos(angle) + Gy*np.sin(angle))
        G_theta_blurred = cv2.GaussianBlur(G_theta, (ksize, ksize), sigma)

        G_thetas.append(G_theta)
        G_thetas_blurred.append(G_theta_blurred)

        # Plot
        if plot:
            ShowImages_Grid([G_theta, G_theta_blurred], 
                nCols=2, 
                titles=[str(round(angle, 2)) + " G_theta", str(round(angle, 2)) + " G_theta_blurred"], 
                figsize=(10, 10), gap=(0.25, 0.25))

    # Calculate CFOG Descriptors
    descriptors = np.dstack(G_thetas_blurred)

    return descriptors

# Runner Functions
def DescriptorGenerate_CFOG(KeypointsData, DescriptorData, params, norm=True):
    '''
    Generates the CFOG descriptors for the given keypoints
    '''
    # Check if no keypoints
    if KeypointsData["Overall_Common_Keypoints"].shape[0] == 0:
        return np.array([])

    sigma = params["CFOG_sigma"]
    ksize = params["CFOG_ksize"]
    N_ORIENTATIONS = params["CFOG_N_ORIENTATIONS"]
    visualise = params["CFOG_visualise"]
    options = params["options"]

    # Get All Descriptors
    I = np.copy(DescriptorData["I"])
    Descriptors_All = GetCFOGDescriptors(I, N_ORIENTATIONS, sigma, ksize, plot=visualise, verbose_main=options["verbose_main"])

    # Get Descriptors of Keypoints
    Descriptors = []
    for i in range(KeypointsData["Overall_Common_Keypoints"].shape[0]):
        Descriptors.append(Descriptors_All[KeypointsData["Overall_Common_Keypoints"][i][0], KeypointsData["Overall_Common_Keypoints"][i][1]])

    Descriptors = np.array(Descriptors)
    
    return Descriptors

# Driver Code
# Params

# Params

# RunCode
print("Reloaded CFOG Descriptor!")