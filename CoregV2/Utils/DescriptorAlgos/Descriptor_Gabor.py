'''
Gabor Descriptor Generation
Reference: https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial74_what%20is%20gabor%20filter.py
'''

# Imports
import numpy as np
import scipy.ndimage
from skimage.filters import gabor_kernel

from ..ImageUtils import *
from ..Normalisers import *

# Main Vars
PARAMS_Gabor = {
    # Kernel Size
    'ksize' : 15, # Use size that makes sense to the image and fetaure size. Large may not be good.

    # Standard Deviation of Gaussian Envelope
    'sigma': 5, # Large sigma on small features will fully miss the features.

    # Orientation of Filter
    'theta': 1*np.pi/2,  # 1/4 shows horizontal 3/4 shows other horizontal. Try other contributions

    # Wavelength of Sine Component
    'lamda': 1*np.pi/4,  # 1/4 works best for angled.

    # Spatial Aspect Ratio
    'gamma': 0.9,  # Value of 1 defines spherical. Calue close to 0 has high aspect ratio
    # Value of 1, spherical may not be ideal as it picks up features from other regions.

    # Phase Offset
    'phi': 0.8  # Phase offset. I leave it to 0. (For hidden pic use 0.8)
}

# Main Functions
# Gabor Functions
def GetGaborDescriptors(I, N_ORIENTATIONS=8, SIGMAS=(1, 5), FREQUENCIES=(0.05, 0.25), plot=False):
    '''
    Get Gabor Descriptors for every pixel in image
    '''

    Angles = np.linspace(0, np.pi, N_ORIENTATIONS, endpoint=False)

    GaborKernels = []
    titles = []
    for theta in Angles:
        for sigma in SIGMAS:
            for frequency in FREQUENCIES:
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                            sigma_x=sigma, sigma_y=sigma))
                GaborKernels.append(kernel)
                titles.append(str(frequency) + "_" + str(np.round(theta, 2)) + "_" + str(sigma))

    # Apply Gabor Filters
    descriptors = np.zeros((I.shape[0], I.shape[1], len(GaborKernels)), dtype=np.float32)
    for k, kernel in enumerate(GaborKernels):
        filtered = scipy.ndimage.convolve(I, kernel, mode='wrap')
        descriptors[:, :, k] = np.copy(filtered)

    # Plot
    if plot:
        kernels_display = [NORMALISERS["MinMax"](np.array(k, dtype=np.float32)) for k in GaborKernels]
        displays = []
        titles_compare = []
        for i in range(len(titles)):
            titles_compare.extend([titles[i], titles[i]])
            displays.extend([kernels_display[i], descriptors[:, :, i]])
            ShowImages_Grid(displays[-2:], 
                            nCols=2, 
                            titles=titles_compare[-2:], 
                            figsize=(5, 5), gap=(0.25, 0.25))

    return descriptors

# Runner Functions
def DescriptorGenerate_Gabor(KeypointsData, DescriptorData, params, norm=True):
    '''
    Generates the Gabor descriptors for the given keypoints
    '''
    # Check if no keypoints
    if KeypointsData["Overall_Common_Keypoints"].shape[0] == 0:
        return np.array([])

    sigmas = params["Gabor_Sigmas"]
    N_ORIENTATIONS = params["Gabor_N_Orientations"]
    frequencies = params["Gabor_Frequencies"]
    visualise = params["Gabor_visualise"]

    # Get All Descriptors
    I = np.copy(DescriptorData["I"])
    Descriptors_All = GetGaborDescriptors(I, N_ORIENTATIONS, sigmas, frequencies, plot=visualise)

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
print("Reloaded Gabor Descriptor!")