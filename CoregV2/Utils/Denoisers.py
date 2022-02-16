'''
Denoisers
'''

# Imports
from .DenoiserAlgos import DeSpeckler_BM3D

# Main Functions


# Main Vars
DENOISERS = {
    "BM3D": DeSpeckler_BM3D.DeSpeckle_BM3D
}

DENOISERS_GRID = {
    "BM3D": DeSpeckler_BM3D.DeSpeckleGrid_BM3D
}

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Denoisers!")