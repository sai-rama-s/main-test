'''
Shift Calculators
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .Normalisers import *
from .ImageUtils import *

# Main Functions
def CalculateShift_AffineTransform(locs_1, locs_2, matchIndices, options=DEFAULT_OPTIONS):
    '''
    Calculates the affine transform between two sets of points
    '''
    options = dict(options)

    # Get Matched Points
    pts_map = []
    for m in matchIndices:
        pts_m = [locs_1[m[0]], locs_2[m[1]]]
        pts_map.append(pts_m)
    pts_map = np.array(pts_map, dtype=np.float32)
    pts_src = pts_map[:, 0]
    pts_dst = pts_map[:, 1]

    # Calculate Affine Transform
    pts_src = np.array(pts_src, dtype=np.float32)
    pts_dst = np.array(pts_dst, dtype=np.float32)
    # H = cv2.getAffineTransform(pts_src, pts_dst)
    H, status = cv2.estimateAffine2D(pts_src, pts_dst)

    return H

def CalculateShift_WeightedMeanShift(locs_1, locs_2, matchIndices, options=DEFAULT_OPTIONS, weights=[]):
    '''
    Calculates the weighted mean shift between two sets of points
    '''
    options = dict(options)

    # Check Empty Weights
    weights = np.array(weights)
    if weights.shape[0] == 0:
        weights = np.ones(matchIndices.shape[0]) / matchIndices.shape[0]

    Shifts = []
    WeightedShifts = []
    for i in range(matchIndices.shape[0]):
        m = matchIndices[i]
        w = weights[i]
        shift = (locs_2[m[1]] - locs_1[m[0]])
        weighted_shift = w * shift
        Shifts.append(list(shift))
        WeightedShifts.append(list(weighted_shift))
    Shifts = np.array(Shifts)
    WeightedShifts = np.array(WeightedShifts)
    OverallShift = np.sum(WeightedShifts, axis=0)

    if options["verbose"]:
        # Display Weights Details
        print("Weights Sum :", np.sum(weights))
        print("Weights Min :", np.min(weights))
        print("Weights Max :", np.max(weights))
        print("Weights Mean:", np.mean(weights))
        print("Weights Std :", np.std(weights))
        # Display Original Shift Details
        print("Shift Min :", np.min(Shifts, axis=0))
        print("Shift Max :", np.max(Shifts, axis=0))
        print("Shift Mean:", np.mean(Shifts, axis=0))
        print("Shift Std :", np.std(Shifts, axis=0))
        # Display Weighted Shift Details
        print("Weighted Shift Sum :", np.sum(WeightedShifts, axis=0))
        print("Weighted Shift Min :", np.min(WeightedShifts, axis=0))
        print("Weighted Shift Max :", np.max(WeightedShifts, axis=0))
        print("Weighted Shift Mean:", np.mean(WeightedShifts, axis=0))
        print("Weighted Shift Std :", np.std(WeightedShifts, axis=0))

    if options["plot"]:
        # Display Weights Histogram
        plt.hist(weights, bins=10)
        plt.title("Weights Histogram")
        plt.show()
        # Display Original Shifts Histogram
        plt.subplot(1, 2, 1)
        plt.hist(Shifts[:, 0], bins=10)
        plt.title("Original X Shifts")
        plt.subplot(1, 2, 2)
        plt.hist(Shifts[:, 1], bins=10)
        plt.title("Original Y Shifts")
        plt.show()
        # Display Weighted Shifts Histogram
        plt.subplot(1, 2, 1)
        plt.hist(WeightedShifts[:, 0], bins=10)
        plt.title("Weighted X Shifts")
        plt.subplot(1, 2, 2)
        plt.hist(WeightedShifts[:, 1], bins=10)
        plt.title("Weighted Y Shifts")
        plt.show()

    # Compute Affine Transform
    H = np.array([
        [1.0, 0.0, -OverallShift[1]],
        [0.0, 1.0, -OverallShift[0]]
    ])

    return H

def CalculateShift_MeanShift(locs_1, locs_2, matchIndices, options=DEFAULT_OPTIONS):
    '''
    Calculates the mean shift between two sets of points
    '''
    options = dict(options)

    # Generate Equal Weights
    weights = np.ones(matchIndices.shape[0]) / matchIndices.shape[0]

    # Calculate H
    H = CalculateShift_WeightedMeanShift(locs_1, locs_2, matchIndices, options=options, weights=weights)

    return H

def CalculateShift_WeightedMeanShift_InverseDistance(locs_1, locs_2, matchIndices, options=DEFAULT_OPTIONS, distances=[]):
    '''
    Calculates the weighted mean shift between two sets of points
    Weights are normalised inverse distances (1 - norm(distance))
    '''
    options = dict(options)

    # Set 0 Distances as the minimum distance
    distances = np.array(distances)
    if np.count_nonzero(distances) > 0:
        distances[distances == 0] = np.min(distances[distances > 0])
    else:
        distances = np.ones(distances.shape)

    # Generate Normalised Inverse Distance Weights (1 - norm(distance))
    weights = 1.0 - NORMALISERS["MinMax"](distances)
    weights = weights / np.sum(weights)

    # Calculate H
    H = CalculateShift_WeightedMeanShift(locs_1, locs_2, matchIndices, options=options, weights=weights)

    return H

def CalculateShift_HistogramShift_TopKDirections(locs_1, locs_2, matchIndices, options=DEFAULT_OPTIONS, distances=[], 
    N_directions=4, K=1, topKHist="weighted", transformMethod="affine"):
    '''
    Calculates transform matrix between two sets of points for top k directions in histogram
    Weights are normalised inverse distances (1 - norm(distance))
    '''
    options = dict(options)

    # Calculate InverseDistance (1 - norm(distance))
    distances = np.array(distances)

    useWeights = True
    if useWeights:
        if np.count_nonzero(distances) > 0:
            distances[distances == 0] = np.min(distances[distances > 0])
        else:
            distances = np.ones(distances.shape)

        weights = 1.0 - NORMALISERS["MinMax"](distances)
    else:
        weights = np.ones(distances.shape)
    weights = weights / np.sum(weights)

    # Calculate Shift Angles for each match
    ShiftMagnitudes = []
    ShiftAngles = []
    for i in range(matchIndices.shape[0]):
        m = matchIndices[i]
        shift_loc = (locs_2[m[1]] - locs_1[m[0]])
        shift = np.array([shift_loc[1], -shift_loc[0]])
        angle = np.arctan2(shift[1], shift[0])
        if angle < 0:
            angle += 2*np.pi # If negative, add 2pi
        ShiftMagnitudes.append(np.sum(shift ** 2) ** (0.5))
        ShiftAngles.append(angle)
    ShiftMagnitudes = np.array(ShiftMagnitudes)
    ShiftAngles = np.array(ShiftAngles)
    # Calculate Shift Directions for each match
    Bins = np.linspace(0, 2*np.pi, N_directions+1)
    BinCenters = (((Bins[:-1] + Bins[1:]) / 2)) % (2*np.pi)
    ShiftDirections = np.digitize(ShiftAngles, Bins) - 1

    # Construct Histogram of Shift Directions
    ShiftDirectionsHistogram = np.zeros(N_directions)
    MagnitudeShiftDirectionsHistogram = np.zeros(N_directions)
    WeightedShiftDirectionsHistogram = np.zeros(N_directions)
    WeightedMagnitudeShiftDirectionsHistogram = np.zeros(N_directions)
    for i in range(matchIndices.shape[0]):
        # print(ShiftDirections[i], ShiftMagnitudes[i], weights[i], ShiftMagnitudes[i] * weights[i])
        m = matchIndices[i]
        shiftDir = ShiftDirections[i]
        ShiftDirectionsHistogram[shiftDir] += 1
        MagnitudeShiftDirectionsHistogram[shiftDir] += ShiftMagnitudes[i]
        WeightedShiftDirectionsHistogram[shiftDir] += weights[i]
        WeightedMagnitudeShiftDirectionsHistogram[shiftDir] += ShiftMagnitudes[i] * weights[i]

    # Intial Clear
    plt.clf()

    # Plot and Save
    plt.figure(figsize=(10, 10))
    polar_ax = plt.subplot(2, 2, 1, projection="polar")
    polar_ax.bar(BinCenters, ShiftDirectionsHistogram, width=np.diff(Bins), bottom=0.0, color="r")  
    for i in range(ShiftDirectionsHistogram.shape[0]):
        polar_ax.text(BinCenters[i], ShiftDirectionsHistogram[i], str(int(ShiftDirectionsHistogram[i])))
    plt.title("Shift Directions Hist")
    polar_ax = plt.subplot(2, 2, 2, projection="polar")
    polar_ax.bar(BinCenters, WeightedShiftDirectionsHistogram, width=np.diff(Bins), bottom=0.0, color="r")
    plt.title("Weighted Shift Directions Hist")
    polar_ax = plt.subplot(2, 2, 3, projection="polar")
    polar_ax.bar(BinCenters, MagnitudeShiftDirectionsHistogram, width=np.diff(Bins), bottom=0.0, color="r")
    plt.title("Magnitude Shift Directions Hist")
    polar_ax = plt.subplot(2, 2, 4, projection="polar")
    polar_ax.bar(BinCenters, WeightedMagnitudeShiftDirectionsHistogram, width=np.diff(Bins), bottom=0.0, color="r")
    plt.title("Weighted Magnitude Shift Directions Hist")

    if options["verbose"]:
        print("Shift Directions Histogram:")
        for i in range(ShiftDirectionsHistogram.shape[0]):
            print(BinCenters[i], "-", int(ShiftDirectionsHistogram[i]), WeightedShiftDirectionsHistogram[i], 
                MagnitudeShiftDirectionsHistogram[i], WeightedMagnitudeShiftDirectionsHistogram[i])

    if options["save"]:
        plt.savefig(options["path"].format("ShiftDirections_Histogram"))

    if options["plot"]:
        plt.show()

    # Final Clear
    plt.clf()

    # Select Top K Directions
    if topKHist == "weighted-magnitude": 
        SelectedHist = WeightedMagnitudeShiftDirectionsHistogram
    elif topKHist == "weighted":
        SelectedHist = WeightedShiftDirectionsHistogram
    elif topKHist == "direction":
        SelectedHist = ShiftDirectionsHistogram
    else:
        SelectedHist = MagnitudeShiftDirectionsHistogram
    if K > SelectedHist.shape[0]:
        K = SelectedHist.shape[0]
    TopKDirections = np.argsort(SelectedHist)[-K:]
    TopKDirections = TopKDirections[::-1]
    # Calculate OverallShift using only these shifts
    updatedMatchIndices = matchIndices[np.isin(ShiftDirections, TopKDirections)]
    weights = weights[np.isin(ShiftDirections, TopKDirections)]
    weights = weights / np.sum(weights)

    # Calculate H
    if transformMethod == "weighted-mean":
        H = CalculateShift_WeightedMeanShift(locs_1, locs_2, updatedMatchIndices, options=options, weights=weights)
    else:
        H = CalculateShift_AffineTransform(locs_1, locs_2, updatedMatchIndices, options=options)

    return H

# Main Vars
SHIFT_COMPUTERS = {
    "AffineTransform": CalculateShift_AffineTransform,
    "MeanShift": CalculateShift_MeanShift,
    "MeanShift_Weighted_InverseDistance": CalculateShift_WeightedMeanShift_InverseDistance,
    "HistogramShift_TopKDirections": CalculateShift_HistogramShift_TopKDirections
}

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Shift Computers!")