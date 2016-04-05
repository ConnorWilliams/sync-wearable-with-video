# Module used for pre processing the accelerometer and video data files.
import numpy as np
import matplotlib.pyplot as plt

rightWrist = 7 + (8*4)
torso = 7 + (9*4)

def readFile(fileName, delimeter):
    return np.genfromtxt(fileName, dtype=float, comments='#', delimiter=delimeter,
                      skip_header=0, skip_footer=0, converters=None, missing_values=None,
                      filling_values=None, usecols=None,
                      names=None, excludelist=None, deletechars=None, replace_space='_',
                      autostrip=True, case_sensitive=True, defaultfmt='f%i',
                      unpack=None, usemask=False, loose=True, invalid_raise=True)

def getTime(ms):
    s = str(ms)
    s = s[:2] + ':' + s[2:4] + ':' + s[4:6] + '.' + s[6:]
    return s

def euclid(*arg):
    total = 0
    for n in arg:
        total += n*n
    return np.sqrt(total)

def extract_acc_section(whole, start_time, end_time, time_idx):
    partial = whole[whole[:,time_idx]>start_time, :]
    partial = partial[partial[:,time_idx]<end_time, :]
    partial = partial[:, 2:6]
    return partial

def extract_vid_section(whole, start_time, end_time, time_idx, joint):
    partial = whole[whole[:,time_idx]>start_time, :]
    partial = partial[partial[:,time_idx]<end_time, :]
    partial = partial[:, np.array([time_idx, joint+1, joint+2, joint+3])]
    return partial

def getSpeed(line1, line2):
    X_dist = line2[1] - line1[1]
    Y_dist = line2[2] - line1[2]
    Z_dist = line2[3] - line1[3]
    time = line2[0]
    return [X_dist/time, Y_dist/time, Z_dist/time]

def getAccel(line1, line2):
    X_dist = line2[1] - line1[1]
    Y_dist = line2[2] - line1[2]
    Z_dist = line2[3] - line1[3]
    time = line2[0]
    tt = time*time
    return [X_dist/tt, Y_dist/tt, Z_dist/tt]

def subsample(bigArray, smallArray):
    bigSize = bigArray.shape[0]
    smallSize = smallArray.shape[0]
    sizeDifference = bigSize - smallSize
    for i in range(0, sizeDifference):
        bigArray = np.delete(bigArray, i, 0)
    return bigArray

def addDrift(signal, mult):
    n = 0
    for l in signal:
        l[0] = l[0] + (n*mult)
        n = n+1
    return signal
