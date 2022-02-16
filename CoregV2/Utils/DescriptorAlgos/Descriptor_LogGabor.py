'''
Log-Gabor Descriptor Generation
Reference: https://python.hotexamples.com/fr/examples/LogGabor/LogGabor/-/python-loggabor-class-examples.html
'''

# Imports
import numpy as np
from .LogGabor import LogGabor

from ..ImageUtils import *
from ..Normalisers import *

# Main Vars
PARAMS_LogGabor = {
    'N_X' : 600, # size of images
    'N_Y' : 600, # size of images
    'noise' : 0.0, # level of noise when we use some
    'do_mask'  : False, # self.pe.do_mask
    # whitening parameters:
    'do_whitening'  : False, 
    # Log-Gabor
    'base_levels' : 1.618,
    'n_theta' : 6, # number of (unoriented) angles between 0. radians (included) and np.pi radians (excluded)
    'B_sf' : .4, # 1.5 in Geisler
    'B_theta' : 3.14159/18.,
    # PATHS
    'use_cache' : True,
    'verbose': 0,
}

# Main Functions
# Log Gabor Functions
def GetLogGaborResponses(I, size=(25, 25), N_SCALES=4, N_ORIENTATIONS=6):
    '''
    Get Log Gabor Responses
    '''

    # Load LogGabor
    PARAMS_LogGabor["N_X"] = I.shape[0]
    PARAMS_LogGabor["N_Y"] = I.shape[1]
    lg = LogGabor(PARAMS_LogGabor)
    # lg.set_size(np.array(size))
    # print(center)

    Responses = []
    Angles = []
    Filters = []
    Titles = []
    for i_level in range(N_SCALES):
        Is_logGabor = []
        Is_logGaborAngles = []
        Is_filters = []
        titles = []
        for theta in np.linspace(0, np.pi, N_ORIENTATIONS, endpoint=False):
            params = {'sf_0':1./(2**i_level), 'B_sf':lg.pe.B_sf, 'theta':theta, 'B_theta':lg.pe.B_theta}
            # loggabor takes as args: u, v, sf_0, B_sf, theta, B_theta)
            FT_lg = lg.loggabor(0, 0, **params)
            # print(FT_lg)

            # Resize I to fit FT_lg
            sameSize = (I.shape[0] == FT_lg.shape[0]) and (I.shape[1] == FT_lg.shape[1])
            if not sameSize:
                I_resized = cv2.resize(I, (FT_lg.shape[1], FT_lg.shape[0]))
            else:
                I_resized = np.copy(I)
            # I_resized = np.copy(I)

            I_logGabor = lg.FTfilter(I_resized, FT_lg, full=True)
            I_logGabor_mag = np.absolute(I_logGabor)
            I_logGabor_angle = np.angle(I_logGabor)

            # Resize Responses
            sameSize = (I.shape[0] == I_logGabor_mag.shape[0]) and (I.shape[1] == I_logGabor_mag.shape[1])
            if not sameSize:
                I_logGabor_mag = cv2.resize(I_logGabor_mag, (I.shape[1], I.shape[0]))
                I_logGabor_angle = cv2.resize(I_logGabor_angle, (I.shape[1], I.shape[0]))

            Is_filters.append(np.absolute(FT_lg))
            Is_logGabor.append(I_logGabor_mag)
            Is_logGaborAngles.append(I_logGabor_angle)
            titles.append(str(i_level) + " " + str(np.round(theta, 2)))
        
        Filters.append(Is_filters)
        Responses.append(Is_logGabor)
        Angles.append(Is_logGaborAngles)
        Titles.append(titles)

    LogGaborData = {
        "Responses": Responses,
        "Angles": Angles,
        "Filters": Filters,
        "Titles": Titles
    }

    for k in LogGaborData.keys():
        LogGaborData[k]= np.array(LogGaborData[k])

    return LogGaborData

# Display Functions
def DisplayLogGaborResponses(I, LogGaborData):
    '''
    Display Log Gabor Responses
    '''
    for i in range(LogGaborData["Titles"].shape[0]):
        for j in range(LogGaborData["Titles"].shape[1]):
            filters = NORMALISERS["MinMax"](LogGaborData["Filters"][i, j])
            responses = NORMALISERS["MinMax"](LogGaborData["Responses"][i, j])
            titles = [LogGaborData["Titles"][i, j], LogGaborData["Titles"][i, j]]
            displays = [filters, responses]
            ShowImages_Grid(displays, 
                            nCols=2, 
                            titles=titles, 
                            figsize=(5, 5), gap=(0.25, 0.25))

# Bin Functions
def LogPolarBinning_LogGabor(keypointData, GradData, 
    N_RADIAL_BINS=3, N_ANGULAR_BINS=8, 
    N_SCALES=4, N_ORIENTATIONS=6, 
    KernelSize=(25, 25), 
    keypoint_centric=True, norm=True):
    '''
    Log Polar Binning with max-based histogram aggregation for Log-Gabor responses
    '''

    # Get Keypoint Data
    x = int(keypointData[0])
    y = int(keypointData[1])
    scale = 1#keypointData[2] # Taken as 1 always since non-linear diffusion scale space is used
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

    # Calculate Log-Gabor Responses around the keypoint
    if keypoint_centric:
        I_keypointSurrounding = GradData["I"][radius_y_up:radius_y_down, radius_x_left:radius_x_right]
        I_surrounding = I_keypointSurrounding

        LogGaborData = GetLogGaborResponses(I_surrounding, KernelSize, 
                                            N_SCALES, N_ORIENTATIONS)
        gradient_surrounding = np.array(LogGaborData["Responses"])
        angle_surrounding = np.array(LogGaborData["Angles"])
        gradient = np.zeros((N_SCALES, N_ORIENTATIONS, GradData["I"].shape[0], GradData["I"].shape[1]))
        gradient[:, :, radius_y_up:radius_y_down, radius_x_left:radius_x_right] = gradient_surrounding
        angle = np.zeros((N_SCALES, N_ORIENTATIONS, GradData["I"].shape[0], GradData["I"].shape[1]))
        angle[:, :, radius_y_up:radius_y_down, radius_x_left:radius_x_right] = angle_surrounding

    # Bin the angles into 6 bins
    ANGLE_BINS = 6

    sub_gradient = gradient[:, :, radius_y_up:radius_y_down, radius_x_left:radius_x_right]
    angle = angle + np.pi
    angle[angle == 2*np.pi] = 0.0
    sub_angle = angle[:, :, radius_y_up:radius_y_down, radius_x_left:radius_x_right]

    sub_gradient_max = np.zeros((sub_gradient.shape[0], sub_gradient.shape[2], sub_gradient.shape[3]), dtype=np.float32)
    sub_angle_max = np.zeros((sub_angle.shape[0], sub_angle.shape[2], sub_angle.shape[3]), dtype=np.float32)
    sub_angle_bins = np.zeros((sub_angle.shape[0], sub_angle.shape[2], sub_angle.shape[3]), dtype=int)

    for s in range(sub_angle.shape[0]):
            # Find Max Gradient Values and corresponding angles
            sub_gradient_max_indices = np.argmax(sub_gradient[s], axis=0)
            for i in range(sub_gradient_max.shape[1]):
                for j in range(sub_gradient_max.shape[2]):
                    sub_gradient_max[s, i, j] = sub_gradient[s, sub_gradient_max_indices[i, j], i, j]
                    sub_angle_max[s, i, j] = sub_angle[s, sub_gradient_max_indices[i, j], i, j]
                    sub_angle_bins[s, i, j] = sub_gradient_max_indices[i, j]

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
                hist_index = sub_angle_bins[:, i, j]
                Mag = sub_gradient_max[:, i, j]

                # Aggregate Histogram based on max Mag of gradient and over all scales (Anyways all hist added over all scales)
                for s in range(sub_angle_bins.shape[0]):
                    index = (amplitude_bin*N_ANGULAR_BINS + angle_bin)*ANGLE_BINS + hist_index[s]
                    temp_hist[int(index)] += Mag[s]

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
def DescriptorGenerate_LogGabor(KeypointsData, DescriptorData, params, norm=True):
    '''
    Generates the Log-Gabor descriptors for the given keypoints
    '''
    # Check if no keypoints
    if KeypointsData["Overall_Common_Keypoints"].shape[0] == 0:
        return np.array([])

    sigma_0 = params["LOGGABOR_sigma_0"]
    kRatio = params["LOGGABOR_ratio"]
    N_RADIAL_BINS = params["LOGGABOR_N_RADIAL_BINS"]
    N_ANGULAR_BINS = params["LOGGABOR_N_ANGULAR_BINS"]
    N_SCALES = params["LOGGABOR_N_SCALES"]
    N_ORIENTATIONS = params["LOGGABOR_N_ORIENTATIONS"]
    KernelSize = params["LOGGABOR_kernel_size"]
    keypoint_centric = params["LOGGABOR_keypoint_centric"]
    binning = params["LOGGABOR_binning"]
    plot = params["LOGGABOR_visualise"]
    options = params["options"]

    # If not keypoint centric, then generate descriptors for full image
    if not keypoint_centric or not binning:
        I = np.copy(DescriptorData["I"])
        LogGaborData = GetLogGaborResponses(I, KernelSize, N_SCALES, N_ORIENTATIONS)
        DescriptorData["Responses"] = LogGaborData["Responses"]
        DescriptorData["Angles"] = LogGaborData["Angles"]

        if plot:
            DisplayLogGaborResponses(I, LogGaborData)

    Descriptors = []
    for i in tqdm(range(KeypointsData["Overall_Common_Keypoints"].shape[0]), disable=not options["verbose_main"]):
        if binning:
            x, y = KeypointsData["Overall_Common_Keypoints"][i][1], KeypointsData["Overall_Common_Keypoints"][i][0]
            layer = KeypointsData["Overall_Common_Keypoints_Scales"][i]
            keypointData = [x, y, sigma_0*(kRatio**(layer)), layer, 0.0]

            # Calculate Descriptor
            descriptor = LogPolarBinning_LogGabor(keypointData, DescriptorData, N_RADIAL_BINS, N_ANGULAR_BINS, N_SCALES, N_ORIENTATIONS, KernelSize, keypoint_centric, norm)
            Descriptors.append(descriptor)
        else:
            descriptor = DescriptorData["Responses"][:, :, KeypointsData["Overall_Common_Keypoints"][i][0], KeypointsData["Overall_Common_Keypoints"][i][1]]
            descriptor = np.reshape(descriptor, (-1))
            Descriptors.append(descriptor)

    Descriptors = np.array(Descriptors)
    
    return Descriptors

# Driver Code
# Params

# Params

# RunCode
print("Reloaded LogGabor Descriptor!")