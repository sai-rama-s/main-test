'''
Evaluation of Coregistration
'''

# Imports
from .utils import *

# Main Functions
# Shift Test Functions
def ShiftTest_Box(size, N_SHIFTS, SHIFT, EvalFunc):
    I_1_pan = np.zeros((300, 300))
    I_2_gray = np.zeros((300, 300))

    Start = [int(size[0]/2), int(size[1]/2)]
    Size = [int(size[0]/10), int(size[1]/10)]
    N_SHIFTS = N_SHIFTS
    N_SKIP = SHIFT

    I_diff = np.zeros((N_SHIFTS*2+1, N_SHIFTS*2+1))
    I_1_pan[Start[0]:Start[0]+Size[0], Start[1]:Start[1]+Size[1]] = 1.0

    for i in tqdm(range(-N_SHIFTS, N_SHIFTS+1)):
        for j in range(-N_SHIFTS, N_SHIFTS+1):
            Shift = [i*N_SKIP, j*N_SKIP]

            I_2_gray[:, :] = 0.0
            I_2_gray[Start[0]+Shift[0]:Start[0]+Shift[0]+Size[0], Start[1]+Shift[1]:Start[1]+Shift[1]+Size[1]] = 1.0

            diff = EvalFunc(I_1_pan, I_2_gray)

            I_diff[i+N_SHIFTS, j+N_SHIFTS] = diff

    return I_diff

def ShiftTest_Image(I, N_SHIFTS, SHIFT, EvalFunc):
    I_1_pan = I

    N_SHIFTS = N_SHIFTS
    N_SKIP = SHIFT

    I_diff = np.zeros((N_SHIFTS*2+1, N_SHIFTS*2+1))

    for i in tqdm(range(-N_SHIFTS, N_SHIFTS+1)):
        for j in range(-N_SHIFTS, N_SHIFTS+1):
            Shift = [i*N_SKIP, j*N_SKIP]

            I_2_gray = ShiftImage(I_1_pan, Shift)

            diff = EvalFunc(I_1_pan, I_2_gray)

            I_diff[i+N_SHIFTS, j+N_SHIFTS] = diff

    return I_diff

# Apply Evaluation Functions
def Coreg_Evaluate(I_1, I_2, EvalFunc):
    '''
    Evaluate the given Coregistration function
    '''

    # Get the difference
    diff = EvalFunc(I_1, I_2)

    return diff

# Main Vars
EVALUATORS_SHIFT_TEST = {
    "Box": ShiftTest_Box,
    "Image": ShiftTest_Image
}

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Evaluation!")