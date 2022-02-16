'''
Normalisers
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Main Functions
def Noise_Gaussian(I, mean=0, var=1):
    '''
    Applies Gaussian Noise to the image
    '''
    row,col,ch = I.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = I + gauss
    return noisy

def Noise_SaltNPepper(I, sp_ratio=0.5, p=0.01):
    '''
    Applies Salt and Pepper Noise to the image
    '''
    row, col, ch = I.shape
    s_vs_p = sp_ratio
    amount = p
    out = np.copy(I)

    # Salt mode
    num_salt = np.ceil(amount * I.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in I.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* I.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in I.shape]
    out[coords] = 0

    return out

def Noise_Poisson(I):
    '''
    Applies Poisson Noise to the image
    '''
    vals = len(np.unique(I))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(I * vals) / float(vals)
    return noisy

def Noise_Speckle(I, strength=0.9):
    '''
    Applies Speckle Noise to the image
    '''
    row, col, ch = I.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)
    noisy = I + I * gauss * strength
    return noisy
      
# Driver Code
# Params

# Params

# RunCode