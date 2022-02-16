'''
Normalisers
'''

# Imports
import cv2
import numpy as np
from skimage import exposure

# Main Functions
def Normalise_ReScale(I, minVal=0.0, maxVal=1.0):
    '''
    Normalises the image by normalising based on given min and max values
    '''

    I = np.copy(I)
    I = Normalise_MinMax(I)
    I = I * (maxVal - minVal) + minVal
    return I

def Normalise_MinMax(I):
    '''
    Normalises the image by normalising based on Image min and max values
    '''

    I = np.copy(I)
    I_max = np.nanmax(I)
    I_min = np.nanmin(I)
    if I_min == I_max:
        return np.ones(I.shape)
    I = (I - I_min) / (I_max - I_min)
    return I

def Normalise_HistogramNorm(I):
    '''
    Normalises the image by normalising based on mean and std of image
    '''

    I = np.copy(I)
    I[I == 0] = np.nan
    mean = np.nanmean(I)
    std =  2 * np.nanstd(I)
    I[I > mean + std] = mean + std
    I = I - (mean - std)
    I[I < 0] = np.nan
    I = I / np.nanmax(I)
    I[I != I] = 0
    return I

def Normalise_HistogramEq(I):
    '''
    Normalises the image using Histogram Equalization
    '''

    I = np.copy(I)
    I = exposure.equalize_hist(I)
    return I

def Normalise_AdaptiveHistogram(I, clip_limit=0.03):
    '''
    Normalises the image using Adaptive Histogram Equalization
    '''
    
    I = np.copy(I)
    I = Normalise_MinMax(I)
    I = exposure.equalize_adapthist(I, clip_limit=clip_limit)
    return I

def Normalise_MinMax_GaussCorrection(I, c=1, gamma=0.6):
    '''
    Normalises the image by using Min-Max normalisation with Gaussian Correction
    '''

    I = np.copy(I)
    I[I == 0] = np.nan
    I = I - np.nanmin(I)
    I[I < 0] = np.nan
    I = I / np.nanmax(I)
    I[I != I] = 0
    gamma_corrected = c * np.power(I, gamma)
    return gamma_corrected

def Normalise_Digitize(I, bins=[0.0, 0.5, 1.0]):
    '''
    Normalises the image to 0.0 to 1.0 and digitizes it to the given bins
    '''

    bins = np.array(bins)
    I = np.copy(I)
    I = Normalise_MinMax(I)
    I_bins = np.digitize(I, bins)
    I = bins[I_bins - 1]
    return I

def Normalise_HistogramMatching(I, ref=None):
    '''
    Normalises the image using Histogram Matching with a reference image
    '''

    I = np.copy(I)
    ref = np.copy(ref)
    I = exposure.match_histograms(I, ref, multichannel=False)
    return I

# Main Vars
NORMALISERS = {
    "MinMax": Normalise_MinMax,
    "ReScale": Normalise_ReScale,
    "Histogram_Norm": Normalise_HistogramNorm,
    "Histogram_Eq": Normalise_HistogramEq,
    "Adaptive_Histogram": Normalise_AdaptiveHistogram,
    "MinMax_GaussCorrection": Normalise_MinMax_GaussCorrection,
    "Digitize": Normalise_Digitize, 
    "Histogram_Matching": Normalise_HistogramMatching
}

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Normalisers!")