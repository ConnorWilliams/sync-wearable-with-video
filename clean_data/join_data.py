import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pre_processing as pre
import fxn_library as f
import temporal_distortion
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

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
    print "%s + %ds = %s" %(t, interval/1000, newTime)
    return int(newTime)

#---- VARIABLES (Change) ----#
time_interval = 30000 #(milliseconds)
start_time = 143419000
date_dir = "Rec_10_06_15_15_33_04_yangdi_1/"
size_of_kernel = 15

#---- DEFINITIONS (Do not change)----#
end_time = addTime(start_time, time_interval)
acc0_dir = "ACC0.000000/"
acc1_dir = "ACC1.000000/"
dev_dir = "Dev0.000000/"
ts_info_filename = date_dir + dev_dir + 'frameTSinfo0.000000.txt'
ts_info_names = ("imName", "idxColour", "tsColour", "idxDepth", "tsDepth", "diff", "tsRecDvc")
video_filename = date_dir + dev_dir + 'user1.txt'
video_names = ()
acc0_filename = date_dir + acc0_dir + 'ACC_0.000000.txt'
acc1_filename = date_dir + acc1_dir + 'ACC_1.000000.txt'
acc_names = ("idx", "diff", "tsRecDvc", "AccX", "AccY", "AccZ", "tsInternal")


#---- START COMPUTATION ----#
# Read in the files specified above in to numpy arrays.
ts_info = pre.readFile(ts_info_filename, "\t")
video = pre.readFile(video_filename, " ")
acc0 = pre.readFile(acc0_filename, "\t")
acc1 = pre.readFile(acc1_filename, "\t")
print 'Files read successfully...'

minTime = max(ts_info[0,6], acc0[0,2], acc1[0,2])
maxTime = min(ts_info[-1,6], acc0[-1,2], acc1[-1,2])
end_time = min(end_time, maxTime)

# Trim ts_info to accomodate for video running before skeleton is detected
while ts_info[0][0] < video[0][0]:
    ts_info = np.delete(ts_info,0,0)
print 'Resized ts_info to be same size as video...'

# Join ts_info and video at index
video = np.delete(video,0,1)
video = np.hstack((ts_info, video))
print 'Joined ts_info and video...'

print "Data is from %s to %s" %(pre.getTime(minTime), pre.getTime(maxTime))

# Extract the sections we are interested in
acc0 = pre.extract_acc_section(acc0, start_time, end_time)
acc1 = pre.extract_acc_section(acc1, start_time, end_time)
video = pre.extract_vid_section(video, start_time, end_time, pre.joint(8))

acc0 = acc0[acc0[:,0]<video[-1,0]]
acc0 = acc0[acc0[:,0]>video[0,0]]
f_x = interpolate.interp1d(video[:,0], video[:,1])
f_y = interpolate.interp1d(video[:,0], video[:,2])
f_z = interpolate.interp1d(video[:,0], video[:,3])
video = np.vstack((acc0[:,0], f_x(acc0[:,0]), f_y(acc0[:,0]), f_z(acc0[:,0]))).T

# Smooth accelerometer data.
acc0[:,1] = np.convolve(acc0[:,1], np.ones((size_of_kernel,))/size_of_kernel, mode='same')
acc0[:,2] = np.convolve(acc0[:,2], np.ones((size_of_kernel,))/size_of_kernel, mode='same')
acc0[:,3] = np.convolve(acc0[:,3], np.ones((size_of_kernel,))/size_of_kernel, mode='same')

# Zero centre data.
means = np.mean(acc0, axis=0)
acc0[:,1] = acc0[:,1]-means[1]
acc0[:,2] = acc0[:,2]-means[2]
acc0[:,3] = acc0[:,3]-means[3]
means = np.mean(video, axis=0)
video[:,1] = video[:,1]-means[1]
video[:,2] = video[:,2]-means[2]
video[:,3] = video[:,3]-means[3]

# Show data is correlated & clean.
X = 1
Y = 2
Z = 3
component = Y
acceleration = np.vstack((acc0[:,0], acc0[:,component])).T
x_pos = np.vstack((video[:,0], video[:,component])).T



# acceleration = temporal_distortion.constant(acceleration, 0.5)
acceleration = temporal_distortion.linear(acceleration, 0.1)
# acceleration = temporal_distortion.periodic(acceleration, -1, 1, sampling_rate)
# acceleration = temporal_distortion.triangular(acceleration, -1, 1, sampling_rate)

cross_corr, norm_cross_corr = f.x_corr(x_pos[:,1], acceleration[:,1])

_, plotnum = plt.subplots(3, sharex=True)

winSize = 2000
stepSize = 2000

# correct_times is a function of the incorrect drifted times.
# correct_times = f(incorrect times)
f = f.sliding_xcorr(x_pos, acceleration, winSize, stepSize, plotnum[1])

new_times = f(acceleration[:,0])

plotnum[0].set_title("Original Data with triangular distortion.")
plotnum[0].plot(x_pos[:,0], x_pos[:,1])
plotnum[0].plot(acceleration[:,0], acceleration[:,1])
plotnum[0].legend(['calculated acceleration', 'accelerometer'])

string = "Sliding X-Corr with window size %1.1fs and step size %1.1f" % (winSize, stepSize)
plotnum[1].set_title(string)

plotnum[2].set_title("Fixed Data")
plotnum[2].plot(x_pos[:,0], x_pos[:,1])
plotnum[2].plot(new_times, acceleration[:,1])
plotnum[2].legend(['calculated acceleration', 'accelerometer'])
plt.xlabel("time")
plt.show()
