# Module used for pre processing the accelerometer and video data files.
import numpy as np
import matplotlib.pyplot as plt


def readFile(fileName, delimeter):
    return np.genfromtxt(fileName, dtype=float, comments='#', delimiter=delimeter,
                      skip_header=0, skip_footer=0, converters=None, missing_values=None,
                      filling_values=None, usecols=None,
                      names=None, excludelist=None, deletechars=None, replace_space='_',
                      autostrip=True, case_sensitive=True, defaultfmt='f%i',
                      unpack=None, usemask=False, loose=True, invalid_raise=True)

def joint(n): return 7+(n*4)

def getTime(ms):
    s = str(ms)
    return s[:2] + ':' + s[2:4] + ':' + s[4:6] + '.' + s[6:]

def extract_acc_section(whole, start_time, end_time):
    partial = whole[whole[:,2]>start_time, :]
    partial = partial[partial[:,2]<end_time, :]
    partial = partial[:, 2:6]
    return partial

def extract_vid_section(whole, start_time, end_time, joint):
    partial = whole[whole[:,6]>start_time, :]
    partial = partial[partial[:,6]<end_time, :]
    partial = partial[:, np.array([6, joint+1, joint+2, joint+3])]
    return partial
