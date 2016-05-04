# Module used for pre processing the accelerometer and video data files.
import numpy as np
from scipy import interpolate

def readFile(fileName, delimeter):
    return np.genfromtxt(fileName, dtype=float, comments='#', delimiter=delimeter,
                      skip_header=0, skip_footer=0, converters=None, missing_values=None,
                      filling_values=None, usecols=None,
                      names=None, excludelist=None, deletechars=None, replace_space='_',
                      autostrip=True, case_sensitive=True, defaultfmt='f%i',
                      unpack=None, usemask=False, loose=True, invalid_raise=True)

def joint(n): return 7+(n*4)

def pre_process(joint_num, start_time, time_interval, ts_info, video, acc0, acc1):
    minTime = max(ts_info[0,6], acc0[0,2], acc1[0,2])
    maxTime = min(ts_info[-1,6], acc0[-1,2], acc1[-1,2])
    print "Data ranges from %s to %s" %(getTime(minTime), getTime(maxTime))

    end_time = min(addTime(start_time, time_interval), maxTime)

    ## Trim ts_info to accomodate for video running before skeleton is detected
    while ts_info[0][0] < video[0][0]:
        ts_info = np.delete(ts_info,0,0)

    ## Join ts_info and video at index
    video = np.delete(video,0,1)
    video = np.hstack((ts_info, video))

    ## Extract the sections we are interested in
    acc0 = extract_acc_section(acc0, start_time, end_time)
    acc1 = extract_acc_section(acc1, start_time, end_time)
    video = extract_vid_section(video, start_time, end_time, joint(joint_num))
    print "Extracted data from %s to %s" %(getTime(start_time), getTime(end_time))
    print "Accelerometer samples at %dHz\nVideo camera samples at %dHz" %(acc0.shape[0]/(time_interval/1000), video.shape[0]/(time_interval/1000))

    ## Interpolate to compensate for different sampling rates.
    acc0 = acc0[acc0[:,0]<video[-1,0]]
    acc0 = acc0[acc0[:,0]>video[0,0]]
    f_x = interpolate.interp1d(video[:,0], video[:,1])
    f_y = interpolate.interp1d(video[:,0], video[:,2])
    f_z = interpolate.interp1d(video[:,0], video[:,3])
    video = np.vstack((acc0[:,0], f_x(acc0[:,0]), f_y(acc0[:,0]), f_z(acc0[:,0]))).T

    from sklearn.preprocessing import StandardScaler
    accel_norm = StandardScaler()
    acc0[:, 1:] = accel_norm.fit_transform(acc0[:, 1:])
    video_norm = StandardScaler()
    video[:, 1:] = video_norm.fit_transform(video[:, 1:])

    ## Smooth accelerometer data.
    acc0[:,1] = np.convolve(acc0[:,1], np.ones((15))/15, mode='same')
    acc0[:,2] = np.convolve(acc0[:,2], np.ones((15))/15, mode='same')
    acc0[:,3] = np.convolve(acc0[:,3], np.ones((15))/15, mode='same')
    return video, acc0, acc1


def addTime(t, interval):
    ms = s = m = h = 0
    s, ms = divmod(interval, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    t = str(t)
    th = int(t[:2])+h
    tm = int(t[2:4])+m
    ts = int(t[4:6])+s
    tms = int(t[6:])+ms
    ms = s = m = h = 0
    s, tms = divmod(tms, 1000)
    ts = ts+s
    m, ts = divmod(ts, 60)
    tm = tm+m
    h, tm = divmod(tm, 60)
    th = th+h
    newTime = str(th)+str(tm)+str(ts)+str(format(tms,'03'))
    return int(newTime)

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
