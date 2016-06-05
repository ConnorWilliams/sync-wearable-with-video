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
distortions_dict = { 0:"No Distortion", 1:"Offset",
                2:"Linear Distortion", 3:"Periodic Distortion",
                4:"Triangular Distortion" }

omega = 0.8
eta = 0.1
distortions = [3]
amount = 0.7

tau=3.2
start_time = 0
end_time = 30
sampling_rate = 50

pns=[0.000001, 0.05, 0.1, 0.2, 0.3, 0.5]
wgns=[0.000001, 10, 100, 200]

for distortion in distortions:
    for pn in pns:
        for wgn in wgns:
            winSize = omega*tau
            stepSize = winSize*eta

            v, a = f.generateData(wgn, pn, end_time, 875, 1/tau, sampling_rate)

            ## Distort the accelerometer data
            a_d = np.copy(a)
            if distortion==2: a_d = temporal_distortion.linear(a_d, amount*tau)
            if distortion==3: a_d = temporal_distortion.periodic(a_d, amount*tau/5)
            if distortion==4: a_d = temporal_distortion.triangular(a_d, int(round(tau/5, -2)))
            if distortion==1: a_d = temporal_distortion.constant(a_d, amount*tau)

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
            a_f = np.vstack((v[:,0], f_n(v[:,0]))).T

            v   =  v[np.logical_not(np.isnan(a_f[:,1]))]
            a   =  a[np.logical_not(np.isnan(a_f[:,1]))]
            a_d =  a_d[np.logical_not(np.isnan(a_f[:,1]))]
            a_f =  a_f[np.logical_not(np.isnan(a_f[:,1]))]

            mse_drifted = ((a[:,1]-a_d[:,1])**2).mean(axis=0)
            mse_fixed = ((a[:,1]-a_f[:,1])**2).mean(axis=0)
            P = mse_fixed/mse_drifted
            if distortion==0: P = mse_fixed
            print "wgn=%1.1f, pn=%1.1f: \t%f" %(wgn, pn, P)

            ## Plot the distorted data
            plt.figure(1)
            plot2 = plt.subplot(211)
            plot2.set_title("Original Data vs Distorted Data")
            ln1 = plot2.plot(v[:,0], v[:,1], 'b', label="$v$", alpha=0.1)
            ln2 = plot2.plot(a[:,0], a[:,1], 'g', label="$a$", alpha=0.6)
            ln3 = plot2.plot(a_d[:,0], a_d[:,1], 'r', label="$a_d$", alpha=0.8)
            lines = ln1+ln2+ln3
            labels = [l.get_label() for l in lines]
            plot2.legend(lines, labels, loc=2, prop={'size':15})

            ## Plot fixed data.
            plt.figure(1)
            # plt.figure(1).suptitle("Data set %s from %s to %s using the %s. MSE(ground truth, drifted)=%f, MSE(ground truth, fixed)=%f"
                # %(date_dir, start_time, end_time, joints[joint_number], mse_drifted, mse_fixed) )
            plot1 = plt.subplot(212)
            plot1.set_title("Original Data vs Fixed Data")
            ln1 = plot1.plot(v[:,0], v[:,1], 'b', label="$v$", alpha=0.1)
            ln2 = plot1.plot(a[:,0], a[:,1], 'g', label="$a$", alpha=0.6)
            ln3 = plot1.plot(a_f[:,0], a_f[:,1], 'r', label="$a_f$", alpha=0.8)
            lines = ln1+ln2+ln3
            labels = [l.get_label() for l in lines]
            plot1.legend(lines, labels, loc=2, prop={'size':15})
            # plt.show()
        print
