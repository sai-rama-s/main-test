'''
GLOH Descriptor Generation
'''

# Imports
import cv2
import numpy as np

from ..ImageUtils import *
from ..Normalisers import *

# Main Functions
# Bin Functions
def LogPolarBinning_GLOH(keypointData, GradData, N_RADIAL_BINS=3, N_ANGULAR_BINS=8, norm=True):
    '''
    Log Polar Binning with histogram aggregation for GLOH responses
    '''

    # Get Keypoint Data
    x = int(keypointData[0])
    y = int(keypointData[1])
    scale = 1#keypointData[2] # Taken as 1 since non-linear diffusion scale space is used
    layer = int(keypointData[3])
    main_angle = keypointData[4]
    gradient = GradData["Responses"]
    angle = GradData["Angles"]

    # Calculate Radius
    cos_t = math.cos(-main_angle/180.*math.pi)
    sin_t = math.sin(-main_angle/180.*math.pi)

    M, N = GradData["I"].shape
    radius = int(round(min(15*scale, min(M/2,N/2))))
    
    radius_x_left = max(0, x - radius)
    radius_x_right = min(N, x + radius + 1)
    radius_y_up = max(0, y - radius)
    radius_y_down = min(M, y + radius + 1)
    
    center_x = x - radius_x_left
    center_y = y - radius_y_up

    # Bin the angles into 8 bins
    ANGLE_BINS = 8

    sub_gradient = gradient[radius_y_up:radius_y_down, radius_x_left:radius_x_right]
    angle = angle + np.pi
    angle[angle == 2*np.pi] = 0.0
    sub_angle = angle[radius_y_up:radius_y_down, radius_x_left:radius_x_right]

    bins = np.arange(0.0, 2*np.pi, 2*np.pi/ANGLE_BINS)
    sub_angle_bins = np.digitize(sub_angle[:, :], bins=bins) - 1

    # Make MeshGrid and construct log-Polar space
    X = list(range(-(x - radius_x_left), radius_x_right - x, 1)) 
    Y = list(range(-(y - radius_y_up), radius_y_down - y, 1))
    [XX,YY] = np.meshgrid(X, Y)
    c_rot = XX*cos_t - YY*sin_t
    r_rot = XX*sin_t + YY*cos_t

    # Handle rotation of keypoint
    log_angle = np.arctan2(r_rot, c_rot)
    log_angle = (log_angle / math.pi) * 180.0
    log_angle[log_angle < 0] = log_angle[log_angle < 0] + 360
    np.seterr(divide='ignore')
    log_amplitude = np.log2(np.sqrt(np.square(c_rot) + np.square(r_rot)))
    
    # Bin the points based on angle
    log_angle = np.round(log_angle * N_ANGULAR_BINS / 360.0)
    log_angle[log_angle <= 0] = log_angle[log_angle <= 0] + N_ANGULAR_BINS
    log_angle[log_angle >= N_ANGULAR_BINS] = log_angle[log_angle >= N_ANGULAR_BINS] - N_ANGULAR_BINS

    # Bin the points based on radius
    r1 = math.log(radius * 0.25, 2)
    r2 = math.log(radius * 0.73, 2)
    log_amplitude[log_amplitude <= r1] = 0
    log_amplitude[(log_amplitude > r1) * (log_amplitude <= r2)] = 1
    log_amplitude[log_amplitude > r2] = 2

    temp_hist = np.zeros(((N_RADIAL_BINS * N_ANGULAR_BINS) * ANGLE_BINS, 1))
    row, col = log_angle.shape
    for i in range(row):
        for j in range(col):
            if (((i-center_y)**2) + ((j-center_x)**2)) <= (radius**2): # Prune out corner points outside radius
                angle_bin = log_angle[i, j]
                amplitude_bin = log_amplitude[i, j]
                hist_index = sub_angle_bins[i, j]
                Mag = sub_gradient[i, j]
                index = (amplitude_bin*N_ANGULAR_BINS + angle_bin)*ANGLE_BINS + hist_index

                # Aggregate the histogram
                temp_hist[int(index)] += Mag

    temp_hist = temp_hist.reshape(-1)

    # Norm
    if norm:
        temp_hist = NORMALISERS["MinMax"](temp_hist)

    # Normalise Histogram
    # temp_hist = temp_hist / np.sqrt(np.sum(temp_hist ** 2))
    # temp_hist[temp_hist>0.2] = 0.2
    # temp_hist = temp_hist / np.sqrt(np.sum(temp_hist ** 2))
    descriptor = temp_hist

    return descriptor

# Runner Functions
def DescriptorGenerate_GLOH(KeypointsData, DescriptorData, params, norm=True):
    '''
    Generates the GLOH descriptors for the given keypoints
    '''
    # Check if no keypoints
    if KeypointsData["Overall_Common_Keypoints"].shape[0] == 0:
        return np.array([])

    sigma_0 = params["GLOH_sigma_0"]
    kRatio = params["GLOH_ratio"]
    N_RADIAL_BINS = params["GLOH_N_RADIAL_BINS"]
    N_ANGULAR_BINS = params["GLOH_N_ANGULAR_BINS"]
    options = params["options"]

    # Generate Sobel Gradients if not given
    if DescriptorData["Responses"] is None:
        I = np.copy(DescriptorData["I"])
        Gx = cv2.Sobel(I, 6, 1, 0)
        Gy = cv2.Sobel(I, 6, 0, 1)
        DescriptorData["Responses"] = np.sqrt(np.square(Gx) + np.square(Gy))
        DescriptorData["Angles"] = np.arctan2(Gy, Gx)

    Descriptors = []
    for i in tqdm(range(KeypointsData["Overall_Common_Keypoints"].shape[0]), disable=not options["verbose_main"]):
        x, y = KeypointsData["Overall_Common_Keypoints"][i][1], KeypointsData["Overall_Common_Keypoints"][i][0]
        layer = KeypointsData["Overall_Common_Keypoints_Scales"][i]
        keypointData = [x, y, sigma_0*(kRatio**(layer)), layer, 0.0]

        # Calculate Descriptor
        descriptor = LogPolarBinning_GLOH(keypointData, DescriptorData, N_RADIAL_BINS, N_ANGULAR_BINS, norm)
        Descriptors.append(descriptor)

    Descriptors = np.array(Descriptors)
    
    return Descriptors

# Driver Code
# Params

# Params

# RunCode
print("Reloaded GLOH Descriptor!")