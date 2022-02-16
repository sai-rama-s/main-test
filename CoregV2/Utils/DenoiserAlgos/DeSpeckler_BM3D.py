'''
DCT Despeckling Low Pass Filter
'''

# Imports
import bm3d

from ..ImageUtils import *
from ..Normalisers import *

# Main Functions
def DeSpeckle_BM3D(I, sigma_psd=30/255, plot=False):
    # Apply BM3D
    I_denoised = bm3d.bm3d(I, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)

    # Plot
    if plot:
        ShowImages_Grid([I, I_denoised], nCols=5, 
                        titles=["I", "I_denoised"], 
                        figsize=(15, 15), gap=(0.25, 0.25))

    return I_denoised

def DeSpeckleGrid_BM3D(I, sigma_psds=[30/255]):
    Is = []
    titles = []
    i = 0
    for sigma_psd in tqdm(sigma_psds):
        i += 1
        I_denoised = DeSpeckle_BM3D(I, sigma_psd=sigma_psd)
        Is.append(I_denoised)
        titles.append(str(i) + ": " + str(sigma_psd))

    # Plot
    cols = 5
    for curInd in range(0, len(Is), cols):
        ShowImages_Grid(Is[curInd:min(curInd+cols, len(Is))], nCols=cols, titles=titles[curInd:min(curInd+cols, len(Is))], figsize=(25, 25), gap=(0.25, 0.25))

    return Is

# Driver Code
# Params

# Params

# RunCode
print("Reloaded BM3D Denoiser!")