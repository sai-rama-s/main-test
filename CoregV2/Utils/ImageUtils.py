'''
Utils
'''

# Imports
#from ssl import HAS_SSLv2
import cv2
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
from tqdm import tqdm

from .Normalisers import NORMALISERS

# Main Vars
DEFAULT_OPTIONS = {
    "verbose_main": True,
    "verbose": True,
    "plot": True,
    "save": False,
    "path": "{}.png",
}

# Main Functions
# Image Functions
def ResizeImage(I, size):
    '''
    Resizes an image to the given size.
    '''
    return cv2.resize(I, size)

def CropBands(I, start, size, ignore3D=False):
    '''
    Crops a image or set of bands from the given start location with specified size.
    '''
    if size[0] <= 0 or size[1] <= 0:
        end = [I.shape[-2], I.shape[-1]]
    else:
        end = [min(start[0] + size[0], I.shape[-2]), min(start[1] + size[1], I.shape[-1])]
    if ignore3D:
        return I[start[0]:end[0], start[1]:end[1]]
    return  I[:, start[0]:end[0], start[1]:end[1]]

def ReadImage(path, size=None):
    '''
    Reads an image from the given path.
    '''
    I = cv2.imread(path)
    if size is not None:
        I = ResizeImage(I, size)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    return I

def SaveImage(I, path):
    '''
    Saves the given image to the given path.
    '''
    I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    I = cv2.imwrite(path, I)

# Conversion Functions
def ClipImageValues(I, minVal=0.0, maxVal=1.0):
    '''
    Clips the given image to the given min and max values.
    '''
    return np.clip(I, minVal, maxVal)

def GetGrayscale(I):
    '''
    Converts the given image to grayscale.
    '''
    return cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

def GetCv2KeyPoints(locs, posOnly=False):
    '''
    Converts the given locations to cv2 keypoints.
    '''
    keypoints = []
    for loc in locs:
        keypoint = cv2.KeyPoint()
        keypoint.pt = (float(loc[0]), float(loc[1]))
        if not posOnly:
            keypoint.octave = int(loc[3])
            keypoint.size = loc[2]
            keypoint.response = loc[5]
        keypoints.append(keypoint)
    return keypoints

# Image Noise Functions
def PSNR(I1, I2):
    '''
    Calculates the Peak Signal to Noise Ratio between two images.
    '''
    return 10 * math.log10(1 / np.mean((I1.ravel() - I2.ravel()) ** 2))

def CSNR(I1, I2):
    '''
    Calculates the Compressed Signal to Noise Ratio between two images.
    '''
    return PSNR(I1, I2) / math.log(2)

def SNR(a):
    '''
    Calculates the Signal to Noise Ratio of the given image.
    '''
    a = a.ravel()
    m = a.mean(0)
    sd = a.std(axis=0, ddof=0)
    return np.where(sd == 0, 0, m / sd)

def ENL(I):
    '''
    Calculates the Equivalent Number of Looks of the given image.
    '''
    return np.mean(I)**2 / np.var(I)

def EPI(I_noisy, I_clean):
    '''
    Calculates the Edge Preservation Index between the given noisy and clean images.
    '''
    EdgeMap_noisy = cv2.Sobel(I_noisy, 6, 1, 1, ksize=3)
    EdgeMap_clean = cv2.Sobel(I_clean, 6, 1, 1, ksize=3)
    EdgeMap_noisy_norm = NORMALISERS["MinMax"](EdgeMap_noisy)
    EdgeMap_clean_norm = NORMALISERS["MinMax"](EdgeMap_clean)
    epi = PSNR(EdgeMap_noisy_norm, EdgeMap_clean_norm)
    return epi

# Display Functions
def ShowImage(I, gray=False, title="", options=DEFAULT_OPTIONS):
    '''
    Displays the given image.
    '''

    # Intial Clear
    plt.clf()

    if gray:
        plt.imshow(I, 'gray')
    else:
        plt.imshow(I)
    plt.title(title)

    # Plot or Save
    if options["save"]:
        plt.savefig(options["path"], bbox_inches='tight')
    if options["plot"]:
        plt.show()

    # Final Clear
    plt.clf()

def ShowImages_Grid(Is, nCols=2, titles=[], figsize=(10, 10), gap=(0.25, 0.25), options=DEFAULT_OPTIONS):
    '''
    Displays the given images in a grid.
    '''
    if not options["save"] and not options["plot"]:
        return

    # Intial Clear
    plt.clf()
    
    nRows = int(math.ceil(len(Is) / nCols))
    plt.figure(figsize=figsize)
    for i in range(len(Is)):
        plt.subplot(nRows, nCols, i+1)
        if Is[i].ndim == 2:
            plt.imshow(Is[i], cmap='gray')
        else:
            plt.imshow(Is[i])
        plt.title(titles[i])
    plt.subplots_adjust(wspace=gap[0], hspace=gap[1])

    # Plot or Save
    if options["save"]:
        plt.savefig(options["path"], bbox_inches='tight')
    if options["plot"]:
        plt.show()

    # Final Clear
    plt.close()
    plt.clf()

def PlotImageHistogram(I, N_BINS=100, title="", options=DEFAULT_OPTIONS):
    '''
    Plots the histogram of the given image.
    '''

    # Intial Clear
    plt.clf()

    plt.hist(I.ravel(), N_BINS)
    plt.title(title)

    # Plot or Save
    if options["save"]:
        plt.savefig(options["path"], bbox_inches='tight')
    if options["plot"]:
        plt.show()
    
    # Final Clear
    plt.clf()

def PlotDescriptors(descriptors, figsize=(10, 10), title="", options=DEFAULT_OPTIONS):
    '''
    Plots the given descriptors.
    '''
    options = dict(options)

    # Initial Clear
    plt.clf()

    # Normalise each descriptor separately
    descriptors_norm = []
    for descriptor in descriptors:
        descriptor_norm = NORMALISERS["MinMax"](descriptor)
        descriptors_norm.append(descriptor_norm)
    
    # Draw Image
    plt.figure(figsize=figsize)
    plt.imshow(descriptors_norm)
    plt.title(title)
    
    # Plot or Save
    if options["save"]:
        plt.savefig(options["path"].format("Descriptors"), bbox_inches='tight')
    if options["plot"]:
        plt.show()

    # Final Clear
    plt.clf()

def PlotDistanceMatrix(DistMatrix, figsize=(10, 10), title="", options=DEFAULT_OPTIONS):
    '''
    Plots the given distance matrix.
    '''
    options = dict(options)

    # Initial Clear
    plt.clf()

    # Normalise distances ignoring np.inf
    DistMatrix_norm = np.empty(DistMatrix.shape)
    DistMatrix_norm[:] = np.nan
    DistMatrix_ValidValuesMask = DistMatrix != np.inf
    DistMatrix_norm[DistMatrix_ValidValuesMask] = NORMALISERS["MinMax"](DistMatrix[DistMatrix_ValidValuesMask])

    # Set Color Map for NaN values
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='purple')
    
    # Draw Image
    plt.figure(figsize=figsize)
    plt.imshow(DistMatrix_norm)
    plt.title(title)
    
    # Plot or Save
    if options["save"]:
        plt.savefig(options["path"].format("Distance_Matrix"), bbox_inches='tight')
    if options["plot"]:
        plt.show()

    # Final Clear
    plt.clf()

# Final Image Generator Functions
def StitchImages(Is, nCols=1):
    '''
    Stitches the given images into a single image.
    '''

    N_Is = len(Is)
    size = (Is[0].shape[0], Is[0].shape[1])
    nRows = int(math.ceil(N_Is / nCols))
    I = np.ones((nRows * size[0], nCols * size[1], 3), dtype=np.uint8) * 255
    imgIndex = 0
    for i in range(nRows):
        for j in range(nCols):
            I[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]] = Is[imgIndex]
            imgIndex += 1
            if imgIndex == N_Is:
                return I
    return I

def ShiftImage(I, Shift):
    '''
    Translates the given image by the given amount.
    '''

    I = np.copy(I)
    Matrix_Translate = np.float32([
        [1, 0, Shift[1]],
        [0, 1, Shift[0]]
    ])
    I_shifted = cv2.warpAffine(I, Matrix_Translate, (I.shape[1], I.shape[0]))
    return I_shifted

def TransformImage(I, H):
    '''
    Transforms the given image by the given affine transform.
    '''

    I = np.copy(I)
    I_transformed = cv2.warpAffine(I, H, (I.shape[1], I.shape[0]))
    return I_transformed

def GetCheckerboardImage(I_1, I_2, resolution=10):
    '''
    Get a checkerboarded image between two images
    '''

    I_checkerboard = np.copy(I_1)

    # False -> I_1
    # True -> I_2
    step = [int(I_1.shape[0]/resolution), int(I_1.shape[1]/resolution)]
    curBlock = [0, 0]
    for i in range(0, I_1.shape[0], step[0]):
        curBlock[1] = 0
        for j in range(0, I_1.shape[1], step[1]):
            if (curBlock[0] + curBlock[1])%2 == 0:
                start = [i, j]
                end = [min(I_1.shape[0], i+step[0]), min(I_1.shape[1], j+step[1])]
                I_checkerboard[start[0]:end[0], start[1]:end[1]] = I_2[start[0]:end[0], start[1]:end[1]]
            curBlock[1] += 1
        curBlock[0] += 1

    return I_checkerboard

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Image Utils!")