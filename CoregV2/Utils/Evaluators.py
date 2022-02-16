'''
Evaluation of Coregistration
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .Normalisers import *
from .ImageUtils import *
from scipy import signal

# Main Functions
# Coregistration Evaluation Functions
def EvalFunc_DFT(I_1, I_2):
    '''
    Evaluates the similarity of two images using DFT
    '''
    dft_1 = cv2.dft(I_1, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_2 = cv2.dft(I_2, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_1 = np.fft.fftshift(dft_1)
    dft_shift_2 = np.fft.fftshift(dft_2)

    magnitude_spectrum_1 = cv2.magnitude(dft_shift_1[:,:,0],dft_shift_1[:,:,1])
    magnitude_spectrum_2 = cv2.magnitude(dft_shift_2[:,:,0],dft_shift_2[:,:,1])

    # magnitude_spectrum_1[magnitude_spectrum_1 != 0.0] = 20*np.log(magnitude_spectrum_1[magnitude_spectrum_1 != 0.0])
    # magnitude_spectrum_2[magnitude_spectrum_2 != 0.0] = 20*np.log(magnitude_spectrum_2[magnitude_spectrum_2 != 0.0])

    diff = np.sum(np.abs(magnitude_spectrum_1 - magnitude_spectrum_2))

    return diff

def EvalFunc_CrossCorrelation(I_1, I_2):
    '''
    Evaluates the similarity of two images using Cross Correlation
    '''
    mag_1 = np.sum(I_1 ** 2) ** (0.5)
    mag_2 = np.sum(I_2 ** 2) ** (0.5)

    diff = np.sum(I_1 * I_2) / (mag_1 * mag_2)
    diff = 1.0 - diff

    return diff

def EvalFunc_RMSE(I_1, I_2):
    '''
    Evaluates the similarity of two images using RMSE
    '''
    diff = np.sum((I_1 - I_2) ** 2) ** (0.5)

    return diff

# Main Vars
EVALUATOR_FUNCTIONS = {
    "DFT": EvalFunc_DFT,
    "CrossCorrelation": EvalFunc_CrossCorrelation,
    "RMSE": EvalFunc_RMSE
}

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Evaluators!")