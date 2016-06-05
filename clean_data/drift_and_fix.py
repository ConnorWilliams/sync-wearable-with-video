import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pre_processing as pre
import fxn_library as f
import temporal_distortion
import sys
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

#---- DEFINITIONS (Do not change)----#
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

segments = [
("Rec_10_06_15_15_33_04_yangdi_1/", 143413000, 27000, 2600),
("Rec_10_06_15_15_50_07_yangdi_2/", 145118000, 18000, 2700),
("Rec_10_06_15_15_50_07_yangdi_2/", 145500000, 17000, 2100),
("Rec_10_06_15_16_38_22_massi_1/", 153917000, 42000, 6000),
("Rec_22_06_15_15_51_08_ben_2/", 145230000, 30000, 4500),
("Rec_22_06_15_16_13_17_jake_1/", 151925000, 16000, 1900)
]

acc0_dir = "ACC0.000000/"
acc1_dir = "ACC1.000000/"
dev_dir = "Dev0.000000/"
ts_info_names = ("imName", "idxColour", "tsColour", "idxDepth", "tsDepth", "diff", "tsRecDvc")
video_names = ()
acc_names = ("idx", "diff", "tsRecDvc", "AccX", "AccY", "AccZ", "tsInternal")

if (len(sys.argv)!=6):
    print("Please give arguments: int joint_number, int component, int windowSize, int stepSize, bool plot")
    exit()

joint_number = int(sys.argv[1])
component = int(sys.argv[2])
omega = float(sys.argv[3])
eta = int(sys.argv[4])
plot = sys.argv[5]
mse_fixed_total = 0

for segment in segments:
    date_dir = segment[0]
    start_time = segment[1]
    time_interval = segment[2]
    tau = segment[3]
    winSize = int(round(omega*tau, -2))
    stepSize = int(round(winSize*eta, -2))

    ts_info_filename = date_dir + dev_dir + 'frameTSinfo0.000000.txt'
    acc0_filename = date_dir + acc0_dir + 'ACC_0.000000.txt'
    acc1_filename = date_dir + acc1_dir + 'ACC_1.000000.txt'
    video_filename = date_dir + dev_dir + 'user1.txt'


    #---- START COMPUTATION ----#
    ## Read in and pre_process the data
    ts_info = pre.readFile(ts_info_filename, "\t")
    video = pre.readFile(video_filename, " ")
    acc0 = pre.readFile(acc0_filename, "\t")
    acc1 = pre.readFile(acc1_filename, "\t")

    # pre.plot_whole(joint_number, ts_info, video, acc0, date_dir)

    video, acc0, acc1, start_time, end_time = pre.pre_process(joint_number, start_time, time_interval, ts_info, video, acc0, acc1)
    print "Data set %s from %s to %s using the %s." %(date_dir, start_time, end_time, joints[joint_number])
    video[:,0] -= video[0,0]
    acc0[:,0] -= acc0[0,0]

    ## Extract the component
    a = np.vstack((acc0[:,0], acc0[:,component])).T
    v = np.vstack((video[:,0], video[:,component])).T


    a_d = np.copy(a)
    ## Distort the accelerometer data
    a_d = temporal_distortion.linear(a_d, tau)
    # a_d = temporal_distortion.periodic(a_d, int(round(tau/5, -2)))
    # a_d = temporal_distortion.triangular(a_d, int(round(tau/5, -2)))
    # a_d = temporal_distortion.constant(a_d, tau)

    ## correct_times is a function of the incorrect drifted times.
    plot3 = plt.subplot(312)
    plot3.set_title("Sliding X-Corr with window size %1.1f and step size %1.1f" % (winSize, stepSize))
    f_t = f.sliding_xcorr(v, a_d, winSize, stepSize, plot3)
    new_times = f_t(a_d[:,0])

    # # Uncomment to plot the time mapping function.
    # plt.figure(2)
    # ln1 = plt.plot((video[0,0],video[-1,0]),(video[0,0],video[-1,0]), 'b', alpha=0.5, label="Ideal time mapping function")
    # ln2 =plt.plot(acceleration[:,0], f(acceleration[:,0]), 'r', label="Time mapping function")
    # lines = ln1+ln2
    # labels = [l.get_label() for l in lines]
    # plt.title("Time mapping functions")
    # plt.xlabel("Drifted times")
    # plt.ylabel("Fixed times")
    # plt.legend(lines, labels, loc=0)

    # Create the fixed signal a_f
    f_n = interpolate.interp1d(new_times, a_d[:,1], bounds_error=False)
    a_f = np.vstack((v[:,0], f_i(v[:,0]))).T



    v   =  v[np.logical_not(np.isnan(a_f[:,1]))]
    a   =  a[np.logical_not(np.isnan(a_f[:,1]))]
    a_d =  a_d[np.logical_not(np.isnan(a_f[:,1]))]
    a_f =  a_f[np.logical_not(np.isnan(a_f[:,1]))]

    mse_drifted = ((a[:,1]-a_d[:,1])**2).mean(axis=0)
    mse_fixed = ((a[:,1]-a_f[:,1])**2).mean(axis=0)
    P = mse_fixed/mse_drifted
    print "MSE(a, a_d)=%f\n" %(mse_drifted)
    print "MSE(a, a_f)=%f\n" %(mse_fixed)
    print "P=%f\n" %(P)

    ## Plot the distorted data
    plt.figure(1)
    plot2 = plt.subplot(211)
    plot2.set_title("Original Data vs Distorted Data")
    ln1 = plot2.plot(v[:,0], v[:,1], 'b', label="position", alpha=0.1)
    ln2 = plot2.plot(a[:,0], a[:,1], 'g', label="original accelerometer data", alpha=0.6)
    ln3 = plot2.plot(a_d[:,0], a_d[:,1], 'r', label="distorted accelerometer data", alpha=0.8)
    lines = ln1+ln2+ln3
    labels = [l.get_label() for l in lines]
    plot2.legend(lines, labels, loc=2)

    ## Plot fixed data.
    plt.figure(1)
    plt.figure(1).suptitle("Data set %s from %s to %s using the %s. MSE(ground truth, drifted)=%f, MSE(ground truth, fixed)=%f"
        %(date_dir, start_time, end_time, joints[joint_number], mse_drift, mse_fixed) )
    plot1 = plt.subplot(212)
    plot1.set_title("Original Data vs Fixed Data")
    ln1 = plot1.plot(v[:,0], v[:,1], 'b', label="position", alpha=0.1)
    ln2 = plot1.plot(a[:,0], a[:,1], 'g', label="original accelerometer data", alpha=0.6)
    ln3 = plot1.plot(a_f[:,0], a_f[:,1], 'r', label="fixed accelerometer data", alpha=0.8)
    lines = ln1+ln2+ln3
    labels = [l.get_label() for l in lines]
    plot1.legend(lines, labels, loc=2)

    plot1.set_xlim([-100, a[-1,0]])
    plot2.set_xlim([-100, a[-1,0]])
    plot3.set_xlim([-100, a[-1,0]])

    if plot=="True": plt.show()
