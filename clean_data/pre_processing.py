# Module used for pre processing the accelerometer and video data files.
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

joints = {
    1: "head",
    2: "neck",
    3: "left shoulder", 4: "right shoulder",
    5: "left elbow",    6: "right elbow",
    7: "left hand",     8: "right hand",
    9: "torso",
    10: "left hip",     11: "right hip",
    12: "left knee",    13: "right knee",
    14: "left foot",    15: "right foot"
}

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
    # print "Data ranges from %s to %s" %(getTime(minTime), getTime(maxTime))

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
    # print "Extracted data from %s to %s" %(getTime(start_time), getTime(end_time))
    # print "Accelerometer samples at %dHz\nVideo camera samples at %dHz" %(acc0.shape[0]/(time_interval/1000), video.shape[0]/(time_interval/1000))

    ## Interpolate to compensate for different sampling rates.
    acc0 = acc0[acc0[:,0]<video[-1,0]]
    acc0 = acc0[acc0[:,0]>video[0,0]]
    f_x = interpolate.interp1d(video[:,0], video[:,1])
    f_y = interpolate.interp1d(video[:,0], video[:,2])
    f_z = interpolate.interp1d(video[:,0], video[:,3])
    video = np.vstack((acc0[:,0], f_x(acc0[:,0]), f_y(acc0[:,0]), f_z(acc0[:,0]))).T

    # Minus mean and divide by std deviation
    from sklearn.preprocessing import StandardScaler
    accel_norm = StandardScaler()
    acc0[:, 1:] = accel_norm.fit_transform(acc0[:, 1:])
    video_norm = StandardScaler()
    video[:, 1:] = video_norm.fit_transform(video[:, 1:])

    ## Calculate absolute acceleration and position
    # abs_acc = np.sqrt(acc0[:,1]**2+acc0[:,2]**2+acc0[:,3]**2)
    # abs_vid = np.sqrt(video[:,1]**2+video[:,2]**2+video[:,3]**2)
    # for i, _ in enumerate(acc0):
    #     acc0[i] = np.append(acc0[i], [abs_acc[i]])
    #     # video[i] = np.append(video[i], abs_vid[i])

    ## Smooth accelerometer data.
    kernelSize = 30
    acc0[:,1] = np.convolve(acc0[:,1], np.ones((kernelSize))/kernelSize, mode='same')
    acc0[:,2] = np.convolve(acc0[:,2], np.ones((kernelSize))/kernelSize, mode='same')
    acc0[:,3] = np.convolve(acc0[:,3], np.ones((kernelSize))/kernelSize, mode='same')
    return video, acc0, acc1, getTime(start_time), getTime(end_time)


def plot_whole(joint_num, ts_info, video, acc0, date_dir):
    minTime = max(ts_info[0,6], acc0[0,2])
    maxTime = min(ts_info[-1,6], acc0[-1,2])
    print "Data ranges from %s to %s" %(getTime(minTime), getTime(maxTime))

    ## Trim ts_info to accomodate for video running before skeleton is detected
    while ts_info[0][0] < video[0][0]:
        ts_info = np.delete(ts_info,0,0)

    ## Join ts_info and video at index
    video = np.delete(video,0,1)
    video = np.hstack((ts_info, video))

    ## Extract the sections we are interested in
    acc0 = extract_acc_section(acc0, minTime, maxTime)
    video = extract_vid_section(video, minTime, maxTime, joint(joint_num))

    print acc0.shape, video.shape

    ## Interpolate to compensate for different sampling rates.
    acc0 = acc0[acc0[:,0]<video[-1,0]]
    acc0 = acc0[acc0[:,0]>video[0,0]]
    f_x = interpolate.interp1d(video[:,0], video[:,1])
    f_y = interpolate.interp1d(video[:,0], video[:,2])
    f_z = interpolate.interp1d(video[:,0], video[:,3])
    video = np.vstack((acc0[:,0], f_x(acc0[:,0]), f_y(acc0[:,0]), f_z(acc0[:,0]))).T

    # Minus mean and divide by std deviation
    from sklearn.preprocessing import StandardScaler
    accel_norm = StandardScaler()
    acc0[:, 1:] = accel_norm.fit_transform(acc0[:, 1:])
    video_norm = StandardScaler()
    video[:, 1:] = video_norm.fit_transform(video[:, 1:])

    ## Smooth accelerometer data.
    kernelSize = 100
    acc0[:,1] = np.convolve(acc0[:,1], np.ones((kernelSize))/kernelSize, mode='same')
    acc0[:,2] = np.convolve(acc0[:,2], np.ones((kernelSize))/kernelSize, mode='same')
    acc0[:,3] = np.convolve(acc0[:,3], np.ones((kernelSize))/kernelSize, mode='same')

    # # Uncomment to plot the whole data.
    plt.figure(2)
    plt.figure(2).suptitle("Data set %s from %s to %s using the %s." %(date_dir, getTime(minTime), getTime(maxTime), joints[joint_num]) )
    ln1 = plt.plot(video[:,0], video[:,3], 'b', label="Video")
    # ax2 = plt.twinx()
    ln2 = plt.plot(acc0[:,0], acc0[:,3], 'r', label="Acceleration")
    lines = ln1+ln2
    labels = [l.get_label() for l in lines]
    # plt.title("Whole data")
    plt.xlabel("Time")
    plt.legend(lines, labels, loc=0)
    plt.show()
    exit()


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
    newTime = str(th)+str(tm)+str(format(ts,'02'))+str(format(tms,'03'))
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
