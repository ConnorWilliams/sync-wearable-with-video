import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pre_processing as pre
import fxn_library as f
import temporal_distortion
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

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


date_dir = "Rec_10_06_15_15_33_04_yangdi_1/"
#---- VARIABLES (Change) ----#
joint_number = 8

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
minTime = 143413000
maxTime = 143437000

print "Data ranges from %s to %s" %(pre.getTime(minTime), pre.getTime(maxTime))

## Trim ts_info to accomodate for video running before skeleton is detected
while ts_info[0][0] < video[0][0]:
    ts_info = np.delete(ts_info,0,0)

## Join ts_info and video at index
video = np.delete(video,0,1)
video = np.hstack((ts_info, video))

true_lag = 300
## Extract the sections we are interested in
original_vid = np.copy(video)
video = pre.extract_vid_section(original_vid, minTime, maxTime, pre.joint(joint_number))
acc0 = pre.extract_vid_section(original_vid, minTime+true_lag, maxTime+true_lag, pre.joint(joint_number))
video[:,0] -= video[0,0]
acc0[:,0] -= acc0[0,0]


from scipy import signal
cross_corr = signal.correlate(video[:,1], acc0[:,1], "full")
cc_ys = np.linspace(-acc0[-1,0], acc0[-1,0], cross_corr.size)
lag_detected = cc_ys[np.argmax(cross_corr)]


plt.figure(1)
plt.figure(1).suptitle("True lag is %dms. Lag detected is %dms" %(true_lag, lag_detected) )
plot1 = plt.subplot(211)
ln1 = plot1.plot(video[:,0], video[:,3], 'b', label="Video")
ln2 = plot1.plot(acc0[:,0], acc0[:,3], 'r', label="Acceleration")
plot2 = plt.subplot(212)
plot2.set_title("Cross-correlation of the two signals.")
ln3 = plot2.plot(cc_ys, cross_corr, 'y', label="Cross-correlation")
# lines = ln1+ln2
# labels = [l.get_label() for l in lines]
# plt.title("Whole data")
plt.xlabel("Time (ms)")
# plt.legend(lines, labels, loc=0)
plt.show()
exit()
