'''
Distance Metrics
'''

# Imports
import numpy as np

from .Normalisers import *
from .ImageUtils import *

# Main Functions
# Distance Calculate Functions
def CalculateDistanceMatrix(descriptors_1, descriptors_2, DistFunc):
    '''
    Calculates the distance matrix between two sets of descriptors
    '''
    DistMatrix = np.empty((descriptors_1.shape[0], descriptors_2.shape[0]))
    DistMatrix[:] = np.inf
    for i in range(descriptors_1.shape[0]):
        for j in range(descriptors_2.shape[0]):
            DistMatrix[i, j] = DistFunc(descriptors_1[i], descriptors_2[j])
    return DistMatrix

# Distance Metric Functions
def Distance_L1(descriptor_1, descriptor_2):
    '''
    Calculates the L1 distance between two descriptors
    '''
    return np.sum(np.abs(descriptor_1 - descriptor_2))

def Distance_L2(descriptor_1, descriptor_2):
    '''
    Calculates the L2 distance between two descriptors
    '''
    return np.sqrt(np.sum(np.square(descriptor_1 - descriptor_2)))

def Distance_NCC(descriptor_1, descriptor_2):
    '''
    Calculates the Normalised Cross Correlation distance between two descriptors
    '''
    descriptor_1_mean = np.mean(descriptor_1)
    descriptor_2_mean = np.mean(descriptor_2)
    descriptor_1_meanshifted = descriptor_1 - descriptor_1_mean
    descriptor_2_meanshifted = descriptor_2 - descriptor_2_mean
    return np.sum(np.multiply(descriptor_1_meanshifted, descriptor_2_meanshifted)) / (np.sqrt(np.sum(np.square(descriptor_1_meanshifted))) * np.sqrt(np.sum(np.square(descriptor_2_meanshifted))))

# Main Vars
DISTANCE_METRICS = {
    "L1": Distance_L1,
    "L2": Distance_L2,
    "NCC": Distance_NCC
}

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Distance Metrics!")