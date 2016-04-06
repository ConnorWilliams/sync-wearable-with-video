import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pre_processing as pre
import programatic
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

#---- VARIABLES (Change) ----#
time_interval = 180000 #(ms)
start_time = 143419000
date_dir = "Rec_10_06_15_15_33_04_yangdi_1/"
size_of_kernel = 15


#---- DEFINITIONS (Do not change)----#
end_time = start_time+time_interval
maxTime = 0
rightWrist = 7 + (8*4)
torso = 7 + (9*4)
acc0_dir = "ACC0.000000/"
acc1_dir = "ACC1.000000/"
dev_dir = "Dev0.000000/"
ts_info_filename = date_dir + dev_dir + 'frameTSinfo0.000000.txt'
ts_info_names = ("imName", "idxColour", "tsColour", "idxDepth", "tsDepth", "diff", "tsRecDvc")
video_filename = top_dir + date_dir + dev_dir + 'user1.txt'
video_names = ()
acc0_filename = top_dir + date_dir + acc0_dir + 'ACC_0.000000.txt'
acc1_filename = top_dir + date_dir + acc1_dir + 'ACC_1.000000.txt'
acc_names = ("idx", "diff", "tsRecDvc", "AccX", "AccY", "AccZ", "tsInternal")


#---- START COMPUTATION ----#
# Read in the files specified above in to numpy arrays.
ts_info = pre.readFile(ts_info_filename, "\t")
video = pre.readFile(video_filename, " ")
acc0 = pre.readFile(acc0_filename, "\t")
acc1 = pre.readFile(acc1_filename, "\t")
print 'Files read successfully...'

maxTime = min(ts_info[-1,6], acc0[-1,2], acc1[-1,2])
end_time = min(start_time+time_interval, maxTime)

# Trim ts_info to accomodate for video running before skeleton is detected
while ts_info[0][0] < video[0][0]:
    ts_info = np.delete(ts_info,0,0)
print 'Resized ts_info to be same size as video...'

# Join ts_info and video at index
video = np.delete(video,0,1)
video = np.hstack((ts_info, video))
print 'Joined ts_info and video...'

# Uncomment this section to generate the skeleton image!!
# skeleton = pre.extract_skeleton(video, start_time, end_time, 6)
# import os, shutil
# folder = 'out'
# for the_file in os.listdir(folder):
#     file_path = os.path.join(folder, the_file)
#     try:
#         if os.path.isfile(file_path):
#             os.unlink(file_path)
#         #elif os.path.isdir(file_path): shutil.rmtree(file_path)
#     except Exception, e:
#         print e
# programatic.makeImage( skeleton )

# Extract the sections we are interested in
acc0 = pre.extract_acc_section(acc0, start_time, end_time, 2)
acc1 = pre.extract_acc_section(acc1, start_time, end_time, 2)
video = pre.extract_vid_section(video, start_time, end_time, 6, rightWrist)

print "video sampling rate =", video.shape[0]/18,"Hz"
print "acc sampling rate =", acc0.shape[0]/18,"Hz"

acc0 = pre.addDrift(acc0, 5)

# Smooth accelerometer data ready for sub sampling.
acc0[:,1] = np.convolve(acc0[:,1], np.ones((size_of_kernel,))/size_of_kernel, mode='same')
acc0[:,2] = np.convolve(acc0[:,2], np.ones((size_of_kernel,))/size_of_kernel, mode='same')
acc0[:,3] = np.convolve(acc0[:,3], np.ones((size_of_kernel,))/size_of_kernel, mode='same')

# Subsample accelerometer data to account for higher frequency.
acc0 = pre.subsample(acc0, video)

if acc0.shape[0] == video.shape[0]:
    print("Smoothed and subsampled accelerometer data...")

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

comp = Y

a = acc0[:,comp]
v = video[:,comp]
na = (a - np.mean(a)) / np.std(a)
nv = (v - np.mean(v)) /  np.std(v)

print 'max( np.correlate(na, nv, "same") ) =\t', max( np.absolute(np.correlate(na, nv, "same")) )
print 'max( np.correlate(a, v, "same") ) =\t', max( np.absolute(np.correlate(a, v, "same")) )

# Plot the data:
plt.title('Timeframe:' + pre.getTime(start_time) + ' - ' + pre.getTime(end_time))
plt.xlabel('time')
line1, = plt.plot(acc0[:,0], na, 'r', label="Accelerometer")
line2, = plt.plot(video[:,0], nv, 'b', label="Video")
plt.gca().twinx()
line3, = plt.plot(video[:,0], np.correlate(na, nv, "same"), 'g', linewidth=2, label="x-corr")
plt.legend([line1, line2, line3],["Accelerometer", "Video", "x-corr"])
plt.show()

# TODO Introduce a drift & fix.
# TODO How do we select that window - the more movement the higher chance of there being correlation? Plot a function or use a ML classifier?
