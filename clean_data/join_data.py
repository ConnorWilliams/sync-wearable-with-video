import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pre_processing as pre
import fxn_library as f
import temporal_distortion
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

#---- VARIABLES (Change) ----#
joint_number = 8
time_interval = 23000 #(milliseconds)
start_time = 143413000
date_dir = "Rec_10_06_15_15_33_04_yangdi_1/"
size_of_kernel = 15

#---- DEFINITIONS (Do not change)----#
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
## Read in and pre_process the data
ts_info = pre.readFile(ts_info_filename, "\t")
video = pre.readFile(video_filename, " ")
acc0 = pre.readFile(acc0_filename, "\t")
acc1 = pre.readFile(acc1_filename, "\t")
video, acc0, acc1 = pre.pre_process(joint_number, start_time, time_interval, ts_info, video, acc0, acc1)

video[:,0] -= video[0,0]
acc0[:,0] -= acc0[0,0]

## Choose a component
X = 1
Y = 2
Z = 3
component = Z

## Extract that component
original_acceleration = np.vstack((acc0[:,0], acc0[:,component])).T
x_pos = np.vstack((video[:,0], video[:,component])).T

acceleration = np.copy(original_acceleration)

## Distort the data
# acceleration = temporal_distortion.constant(acceleration, 500)
acceleration = temporal_distortion.linear(acceleration, 1000)
# acceleration = temporal_distortion.periodic(acceleration, 500)
# acceleration = temporal_distortion.triangular(acceleration, 300)

plt.figure(1)
## Plot the distorted data
plot2 = plt.subplot(311)
plot2.set_title("Original Data vs Distorted Data")
ln1 = plot2.plot(x_pos[:,0], x_pos[:,1], 'b', label="x_pos")
ax2 = plot2.twinx()
ln2 = ax2.plot(original_acceleration[:,0], original_acceleration[:,1], 'g', label="original")
ln3 = ax2.plot(acceleration[:,0], acceleration[:,1], 'r', label="distorted", alpha=0.5)
lines = ln1+ln2+ln3
labels = [l.get_label() for l in lines]
plot2.legend(lines, labels, loc=0)


## Track and fix the distortion
winSize = 2000
stepSize = 500

## correct_times is a function of the incorrect drifted times.
## correct_times = f(incorrect times)
plot3 = plt.subplot(312)
plot3.set_title("Sliding X-Corr with window size %1.1f and step size %1.1f" % (winSize, stepSize))
f = f.sliding_xcorr(x_pos, acceleration, winSize, stepSize, plot3)

plt.figure(2)
ln1 = plt.plot((video[0,0],video[-1,0]),(video[0,0],video[-1,0]), 'b', alpha=0.5, label="Ideal time mapping function")
ln2 =plt.plot(acceleration[:,0], f(acceleration[:,0]), 'r', label="Time mapping function")
lines = ln1+ln2
labels = [l.get_label() for l in lines]
plt.title("Time mapping functions")
plt.xlabel("Drifted times")
plt.ylabel("Fixed times")
plt.legend(lines, labels, loc=0)

new_times = f(acceleration[:,0])

## Plot original unshifter data.
plt.figure(1)
plot1 = plt.subplot(313)
plot1.set_title("Original Data vs Fixed Data")
ln1 = plot1.plot(x_pos[:,0], x_pos[:,1], 'b', label="x_pos")
ax2 = plot1.twinx()
ln2 = ax2.plot(original_acceleration[:,0], original_acceleration[:,1], 'g', label="original")
ln3 = ax2.plot(new_times, acceleration[:,1], 'r', label="fixed")
lines = ln1+ln2+ln3
labels = [l.get_label() for l in lines]
plot1.legend(lines, labels, loc=0)

plt.xlabel("time")
plt.show()
