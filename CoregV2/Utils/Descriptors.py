'''
Denoisers
'''

# Imports
from .DescriptorAlgos import Descriptor_Position, Descriptor_LogGabor, Descriptor_GLOH, Descriptor_CFOG, Descriptor_Gabor

# Main Functions


# Main Vars
DESCRIPTORS = {
    "Position": Descriptor_Position.DescriptorGenerate_Position,
    "LogGabor": Descriptor_LogGabor.DescriptorGenerate_LogGabor,
    "GLOH": Descriptor_GLOH.DescriptorGenerate_GLOH,
    "CFOG": Descriptor_CFOG.DescriptorGenerate_CFOG,
    "Gabor": Descriptor_Gabor.DescriptorGenerate_Gabor
}

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Descriptors!")