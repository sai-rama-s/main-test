'''
Position Descriptor Generation
'''

# Imports
import numpy as np

from ..ImageUtils import *
from ..Normalisers import *

# Main Functions
# Descriptor Functions
def GetPositionDescriptors(KeypointsData, size, norm=False):
    '''
    Gets the position descriptors for the given keypoints
    '''
    # Check if no keypoints
    if KeypointsData["Overall_Common_Keypoints"].shape[0] == 0:
        return np.array([])

    kps = np.array(np.copy(KeypointsData["Overall_Common_Keypoints"]), dtype=np.float32)
    kps[:, 0] = kps[:, 0] / size[0]
    kps[:, 1] = kps[:, 1] / size[1]
    return kps

# Runner Functions
def DescriptorGenerate_Position(KeypointsData, DescriptorData, params, norm=False):
    '''
    Generates the position descriptors for the given keypoints
    '''

    Descriptors = GetPositionDescriptors(KeypointsData, DescriptorData["I"].shape, norm=norm)
    Descriptors = np.array(Descriptors)
    
    return Descriptors

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Position Descriptor!")