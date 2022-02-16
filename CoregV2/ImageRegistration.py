"""
Image Registration
"""

# Imports
import cv2
import numpy as np

from .utils import *

# Main Functions
# Match Functions
def Match_LeastDistance_BestOnly(possibleMatchIndices, descriptors_1, descriptors_2, DistFunc=DISTANCE_METRICS["L2"], options=DEFAULT_OPTIONS):
    '''
    Get Final Matches based on Least Descriptor Distance
    If best match is already taken, current keypoint is ignored
    '''
    # Construct Possible Matches Array
    minDistMatch = []
    distances = []
    DistMatrix = np.empty((descriptors_1.shape[0], descriptors_2.shape[0]))
    DistMatrix[:, :] = np.inf
    for i in range(len(possibleMatchIndices)):
        if len(possibleMatchIndices[i]) == 0:
            continue
        distances_i = np.array([DistFunc(descriptors_1[i], descriptors_2[j]) for j in possibleMatchIndices[i]])
        bestMatchIndex = np.argmin(distances_i)
        bestMatch = possibleMatchIndices[i][bestMatchIndex]
        minDistMatch.append([i, bestMatch])
        distances.append(distances_i[bestMatchIndex])
        for k in range(len(possibleMatchIndices[i])):
            DistMatrix[i, possibleMatchIndices[i][k]] = distances_i[k]

    # Sort Matches by Distance
    sorted_order = np.argsort(distances)
    possibleMatches_sorted = [minDistMatch[i] for i in sorted_order]
    distances_sorted = [distances[i] for i in sorted_order]
    
    # Loop till empty
    matches = []
    distances = []
    if options["verbose_main"]:
        PossibleMatchesCount = len(possibleMatches_sorted)
        pbar = tqdm(total=PossibleMatchesCount)
    while len(possibleMatches_sorted) > 0:
        # Get Least Distance Match and Add to Final Matches
        m = possibleMatches_sorted[0]
        matches.append(m)
        distances.append(distances_sorted[0])
        # Remove all Matches with same keypoints
        possibleMatches_sorted_updated = []
        distances_sorted_updated = []
        for i in range(1, len(possibleMatches_sorted)):
            if possibleMatches_sorted[i][0] != m[0] and possibleMatches_sorted[i][1] != m[1]:
                possibleMatches_sorted_updated.append(possibleMatches_sorted[i])
                distances_sorted_updated.append(distances_sorted[i])
        possibleMatches_sorted = possibleMatches_sorted_updated
        distances_sorted = distances_sorted_updated

        # Update Progress Bar
        if options["verbose_main"]:
            curProgress = PossibleMatchesCount - (len(possibleMatches_sorted))
            pbar.update(curProgress - pbar.n)
    if options["verbose_main"]:
        pbar.close()

    return matches, distances, DistMatrix

def Match_LeastDistance_BestAvailable(possibleMatchIndices, descriptors_1, descriptors_2, DistFunc=DISTANCE_METRICS["L2"], options=DEFAULT_OPTIONS):
    '''
    Get Final Matches based on Least Descriptor Distance on available free points
    If best match is already taken, next available best is taken
    '''
    # Construct Possible Matches Array
    possibleMatches = []
    distances = []
    DistMatrix = np.empty((descriptors_1.shape[0], descriptors_2.shape[0]))
    DistMatrix[:, :] = np.inf
    for i in range(len(possibleMatchIndices)):
        for j in possibleMatchIndices[i]:
            possibleMatches.append([i, j])
            dist = DistFunc(descriptors_1[i], descriptors_2[j])
            distances.append(dist)
            DistMatrix[i, j] = dist

    # Sort Matches by Distance
    sorted_order = np.argsort(distances)
    possibleMatches_sorted = [possibleMatches[i] for i in sorted_order]
    distances_sorted = [distances[i] for i in sorted_order]

    # Loop till empty
    matches = []
    distances = []
    if options["verbose_main"]:
        PossibleMatchesCount = len(possibleMatches_sorted)
        pbar = tqdm(total=PossibleMatchesCount)
    while len(possibleMatches_sorted) > 0:
        # Get Least Distance Match and Add to Final Matches
        m = possibleMatches_sorted[0]
        matches.append(m)
        distances.append(distances_sorted[0])
        # Remove all Matches with same keypoints
        possibleMatches_sorted_updated = []
        distances_sorted_updated = []
        for i in range(1, len(possibleMatches_sorted)):
            if possibleMatches_sorted[i][0] != m[0] and possibleMatches_sorted[i][1] != m[1]:
                possibleMatches_sorted_updated.append(possibleMatches_sorted[i])
                distances_sorted_updated.append(distances_sorted[i])
        possibleMatches_sorted = possibleMatches_sorted_updated
        distances_sorted = distances_sorted_updated

        # Update Progress Bar
        if options["verbose_main"]:
            curProgress = PossibleMatchesCount - (len(possibleMatches_sorted))
            pbar.update(curProgress - pbar.n)
    if options["verbose_main"]:
        pbar.close()

    return matches, distances, DistMatrix

# Template Matching Functions
def ImageRegister_TemplateMatch(keypoints_1, keypoints_2, descriptors_1, descriptors_2, I_1, I_2, 
    ImageRegisterParams, options=DEFAULT_OPTIONS):
    '''
    Image Registration using Template Matching
    '''
    options = dict(options)

    search_window = ImageRegisterParams["search_window"]
    DistFunc = ImageRegisterParams["DistFunc"]
    RANSAC = ImageRegisterParams["RANSAC"]
    MatchType = ImageRegisterParams["MatchType"]

    locs_1 = []
    locs_2 = []
    I_locs_1 = np.zeros(I_1.shape[:2], dtype=bool)
    I_locs_2 = np.zeros(I_1.shape[:2], dtype=bool)
    I_indices_2 = np.ones(I_1.shape[:2], dtype=int) * -1
    for k in keypoints_1:
        l = [int(k.pt[1]), int(k.pt[0])]
        locs_1.append(l)
        I_locs_1[l[0], l[1]] = True
    for ki in range(len(keypoints_2)):
        k = keypoints_2[ki]
        l = [int(k.pt[1]), int(k.pt[0])]
        locs_2.append(l)
        I_locs_2[l[0], l[1]] = True
        I_indices_2[l[0], l[1]] = ki
    locs_1 = np.array(locs_1)
    locs_2 = np.array(locs_2)

    # For every point in Optical, check keypoints in SAR within search_window
    possibleMatchIndices = []
    for i in range(locs_1.shape[0]):
        loc_opt = locs_1[i]
        
        bounds = np.array([
                  [max(0, loc_opt[0]-search_window), min(I_1.shape[0], loc_opt[0]+search_window)], 
                  [max(0, loc_opt[1]-search_window), min(I_1.shape[1], loc_opt[1]+search_window)]
        ])
        I_check_window = I_locs_2[bounds[0, 0]:bounds[0, 1], bounds[1, 0]:bounds[1, 1]]
        nearby_sar_locs = Map2Points(I_check_window) + bounds[:, 0]
        nearby_sar_indices = []
        for l in nearby_sar_locs:
            if I_indices_2[l[0], l[1]] > -1:
                nearby_sar_indices.append(I_indices_2[l[0], l[1]])
        possibleMatchIndices.append(nearby_sar_indices)

    # Get Final Matches
    if MatchType == "BestOnly":
        matchIndices, matchDistances, DistMatrix = Match_LeastDistance_BestOnly(possibleMatchIndices, descriptors_1, descriptors_2, DistFunc, options)
    else:
        matchIndices, matchDistances, DistMatrix = Match_LeastDistance_BestAvailable(possibleMatchIndices, descriptors_1, descriptors_2, DistFunc, options)
    matchIndices = np.array(matchIndices)
    matchDistances = np.array(matchDistances)

    # Display Distance Matrix
    PlotDistanceMatrix(DistMatrix, figsize=(10, 10), title="Distance Matrix", options=options)

    # Get Pts Map
    matches = []
    pts_map = []
    for i in tqdm(range(matchIndices.shape[0]), disable=not options["verbose_main"]):
        opt_index = matchIndices[i][0]
        sar_index = matchIndices[i][1]
        minDist = matchDistances[i]
        loc_opt = locs_1[opt_index]
        loc_sar = locs_2[sar_index]

        match_obj = cv2.DMatch(opt_index, sar_index, minDist)
        matches.append(match_obj)
        pts_map.append([
                        [loc_opt[0], loc_opt[1]],
                        [loc_sar[0], loc_sar[1]]
        ])
    pts_map = np.array(pts_map)

    # Perform RANSAC
    if RANSAC:
        H, status = cv2.findHomography(pts_map[:, 0], pts_map[:, 1], cv2.RANSAC, 1.0)
        ptsMask = np.array(status, dtype=bool).reshape(-1)
        matches_RANSAC = []
        matchIndices_RANSAC = []
        matchDistances_RANSAC = []
        for i in range(ptsMask.shape[0]):
            pmask = ptsMask[i]
            m = matches[i]
            mInd = matchIndices[i]
            mDist = matchDistances[i]
            if pmask:
                matches_RANSAC.append(m)
                matchIndices_RANSAC.append(list(mInd))
                matchDistances_RANSAC.append(mDist)
        matches = matches_RANSAC
        matchIndices = np.array(matchIndices_RANSAC)
        matchDistances = np.array(matchDistances_RANSAC)

    N_MATCHES = len(matches)
    if options["verbose_main"]:
        print("Number of Matches:", N_MATCHES)
        avgDist = np.mean(matchDistances)
        print("Average Match Distance:", avgDist)

        I_matchDistances = np.array(matchDistances).reshape((-1, 1))
        options["path"] = options["path"].format("ImageRegister_MatchDistance_Histogram")
        PlotImageHistogram(I_matchDistances, 10, "Match Distance Histogram", options)
    
    # Draw Final Matches
    I_matched = cv2.drawMatches(I_1, keypoints_1, I_2, keypoints_2, matches[:N_MATCHES], I_2, flags=2)

    return I_matched, matchIndices, matchDistances

# CV2 Matching Functions
def ImageRegister_CV2(keypoints_1, keypoints_2, descriptors_1, descriptors_2, I_1, I_2, 
    ImageRegisterParams, options=DEFAULT_OPTIONS):
    '''
    Image Registration using CV2
    '''
    options = dict(options)
    
    distFunc = ImageRegisterParams["DistFunc_CV2"]
    crossCheck = ImageRegisterParams["crossCheck"]
    RANSAC = ImageRegisterParams["RANSAC"]
    
    # Rectify datatype
    descriptors_1 = np.array(descriptors_1, dtype=np.float32)
    descriptors_2 = np.array(descriptors_2, dtype=np.float32)

    # Create Feature Matcher
    bf = cv2.BFMatcher(distFunc, crossCheck=crossCheck)
    # Match Descriptors of both Images
    matches = bf.match(descriptors_1, descriptors_2)
    # Sort Matches by Distance
    matches = sorted(matches, key = lambda x:x.distance)

    matchIndices = []
    pts_map = []
    for m in matches:
        pts_m_indices = [m.queryIdx, m.trainIdx]
        matchIndices.append(pts_m_indices)
        loc_1 = list(keypoints_1[pts_m_indices[0]].pt)
        loc_2 = list(keypoints_2[pts_m_indices[1]].pt)
        pts_map.append([loc_1, loc_2])
    matchIndices = np.array(matchIndices)
    pts_map = np.array(pts_map)

    # Perform RANSAC
    if RANSAC:
        H, status = cv2.findHomography(pts_map[:, 0], pts_map[:, 1], cv2.RANSAC, 1.0)
        ptsMask = np.array(status, dtype=bool).reshape(-1)
        matches_RANSAC = []
        matchIndices_RANSAC = []
        for i in range(ptsMask.shape[0]):
            pmask = ptsMask[i]
            m = matches[i]
            mInd = matchIndices[i]
            if pmask:
                matches_RANSAC.append(m)
                matchIndices_RANSAC.append(list(mInd))
        matches = matches_RANSAC
        matchIndices = np.array(matchIndices_RANSAC)

    N_MATCHES = len(matches)
    if options["verbose_main"]:
        print("Number of Matches:", N_MATCHES)
        matchDistances = [m.distance for m in matches]
        avgDist = np.mean(matchDistances)
        print("Average Match Distance:", avgDist)

        I_matchDistances = np.array(matchDistances).reshape((-1, 1))
        options["path"] = options["path"].format("ImageRegister_MatchDistance_Histogram")
        PlotImageHistogram(I_matchDistances, 10, "Match Distance Histogram", options)
    
    # Draw Final Matches
    I_matched = cv2.drawMatches(I_1, keypoints_1, I_2, keypoints_2, matches[:N_MATCHES], I_2, flags=2)

    return I_matched, matchIndices, matchDistances

# Display Functions
def GetSearchWindowedImage(I, locs, search_window=1):
    '''
    Get a image with search windows around the given keypoints
    '''

    I_swindow = np.copy(I)
    for loc in locs:
        bounds = np.array([
                  [max(0, loc[0]-search_window), min(I.shape[0], loc[0]+search_window)], 
                  [max(0, loc[1]-search_window), min(I.shape[1], loc[1]+search_window)]
        ])
        I_swindow = cv2.rectangle(I_swindow, tuple(bounds[:, 0][::-1]), tuple(bounds[:, 1][::-1]), (255, 0, 0), 1)
    return I_swindow

def GetKeypointComparisonImage(I, locs_1, keypoints_1, keypoints_2, matchIndices, search_window=1, keypointsDraw=[True, True]):
    '''
    Get a image with keypoints from image 1 and their matches from image 2 with search windows
    '''

    I_comparison = np.copy(I)
    # Mark Optical Keypoints with Green
    if keypointsDraw[0]:
        I_comparison = cv2.drawKeypoints(I_comparison, keypoints_1, 0, (0, 255, 0))
    # Mark Optical Template Windows
    I_comparison = GetSearchWindowedImage(I_comparison, locs_1, search_window)
    # Mark SAR Keypoints with Blue
    if keypointsDraw[1]:
        I_comparison = cv2.drawKeypoints(I_comparison, keypoints_2, 0, (0, 0, 255))
    # Mark Connectors
    for m in matchIndices:
        loc_opt = tuple(np.array(keypoints_1[m[0]].pt, dtype=int))
        loc_sar = tuple(np.array(keypoints_2[m[1]].pt, dtype=int))
        I_comparison = cv2.line(I_comparison, loc_opt, loc_sar, (255, 255, 0))

    return I_comparison

def DisplayCheckerboardComparison(I_1, I_2, I_2_corrected, resolution=4, options=DEFAULT_OPTIONS):
    '''
    Display a checkerboard comparison between two images
    '''
    options = dict(options)
    options_temp = dict(options)

    # Display Checkerboard Image - Original
    options_temp["path"] = options["path"].format("Checkerboard_Original")
    I_checkerboard = GetCheckerboardImage(I_1, I_2, resolution)
    ShowImages_Grid([I_1, I_2, I_checkerboard], 
                    nCols=3, 
                    titles=["Optical", "SAR", "Checkerboard"], 
                    figsize=(20, 20), gap=(0.25, 0.25), 
                    options=options_temp)

    # Display Checkerboard Image - Corrected
    options_temp["path"] = options["path"].format("Checkerboard_Corrected")
    I_checkerboard_corrected = GetCheckerboardImage(I_1, I_2_corrected, resolution)
    ShowImages_Grid([I_1, I_2_corrected, I_checkerboard_corrected], 
                    nCols=3, 
                    titles=["Optical", "Corrected SAR", "Corrected Checkerboard"], 
                    figsize=(20, 20), gap=(0.25, 0.25), 
                    options=options_temp)
    
    # Final Checkered Images
    options_temp["path"] = options["path"].format("Final_Checkerboard_Comparison")
    I_checkerboard_corrected = GetCheckerboardImage(I_1, I_2_corrected, resolution)
    ShowImages_Grid([I_checkerboard, I_checkerboard_corrected], 
                    nCols=2, 
                    titles=["Original Checkerboard", "Corrected Checkerboard"], 
                    figsize=(20, 20), gap=(0.25, 0.25), 
                    options=options_temp)

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Image Registration!")