'''
General Utils
'''

# Imports
import json
import numpy as np

# Main Functions
# General Functions
def FormatTime(secs, precision=4):
    '''
    Formats the given time in seconds to a string.
    '''

    hours, rem = divmod(secs, 3600)
    minutes, seconds = divmod(rem, 60)
    timeStr = ("{:0>2} : {:0>2} : {:05." + str(precision) + "f}").format(int(hours),int(minutes),seconds)
    return timeStr

def GetCountData(counts, bins, target=200):
    '''
    Gets the count data and target threshold for the given counts and bins.
    '''

    targetThreshold = None
    countData = []
    cumulativeCount = int(np.sum(counts))
    for i in range(len(counts)):
        b = bins[i+1]
        c = int(counts[i])
        cumulativeCount -= c
        countData.append([np.round(b, 4), c, cumulativeCount])
        if cumulativeCount <= target and targetThreshold is None:
            targetThreshold = [bins[i], b, cumulativeCount]
        # print(b, ":", c)
    return countData, targetThreshold

def GetFormattedAvailableFunctions(OBJ):
    '''
    Gets the formatted string of available functions for the given functions dict object.
    '''

    return str(json.dumps(list(OBJ.keys()), indent=4))

def Map2Points(Map):
    '''
    Gets locations of True values in the map.
    '''

    X, Y = np.where(Map == True)
    Pts = np.dstack((X, Y))[0]
    return Pts

# Driver Code
# Params

# Params

# RunCode
print("Reloaded General Utils!")