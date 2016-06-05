import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal
import fxn_library as f
import temporal_distortion
import sys
import wave
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)


# if (len(sys.argv)!=3):
#     print("Please give arguments: int windowSize, int stepSize")
#     exit()
# windowSize = float(sys.argv[1])
# stepSize = int(sys.argv[2])
# winSize = windowSize

winSizes = [0.8]
stepSize = 0.5
tau = 3.2

for winSize in winSizes:
        #---- START COMPUTATION ----#
        ## Get some data
        start_time = 0
        end_time = 30
        sampling_rate = 50
        x_pos, acceleration = f.generateData(0.1, 0.00001, end_time, 875, 1/tau, sampling_rate)

        # tau = 5
        # time = np.arange(start_time, end_time, float(1)/sampling_rate)
        # x_pos = signal.square(2*np.pi*time/tau)
        # x_pos = np.vstack((time, x_pos)).T
        # acceleration = np.copy(x_pos)

        # print "Data set is from %s to %s." %(x_pos[0,0], x_pos[-1,0])
        # print "Window Size = %f, Step Size = %f" %(winSize, stepSize)

        ## Distort the data
        # acceleration = temporal_distortion.linear(acceleration, tau)
        acceleration = temporal_distortion.periodic(acceleration, tau/5)
        # acceleration = temporal_distortion.triangular(acceleration, tau)
        # acceleration = temporal_distortion.constant(acceleration, tau)

        mse_drift = ((x_pos[:,1] - acceleration[:,1])**2).mean(axis=0)

        ## correct_times is a function of the incorrect drifted times.
        plot3 = plt.subplot(312)
        plot3.set_title("Sliding X-Corr with window size %1.1f and step size %1.1f" % (winSize, stepSize))
        f_t = f.sliding_xcorr(x_pos, acceleration, winSize*tau, stepSize, plot3)


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

        new_times = f_t(acceleration[:,0])
        f_i = interpolate.interp1d(new_times, acceleration[:,1], bounds_error=False)
        fixed_acceleration = np.vstack((x_pos[:,0], f_i(x_pos[:,0]))).T

        x_pos =  x_pos[np.logical_not(np.isnan(fixed_acceleration[:,1]))]
        # acceleration =  acceleration[np.logical_not(np.isnan(fixed_acceleration[:,1]))]
        new_times = new_times[np.logical_not(np.isnan(fixed_acceleration[:,1]))]
        fixed_acceleration =  fixed_acceleration[np.logical_not(np.isnan(fixed_acceleration[:,1]))]

        mse_fixed = ((x_pos[:,1] - fixed_acceleration[:,1])**2).mean(axis=0)
        # print "MSE between ground truth and drifted is %f" %(mse_drift)
        # print "MSE between ground truth and fixed is %f\n" %(mse_fixed)
        print "%f\n" %(mse_fixed)

        ## Plot the distorted data
        plt.figure(1)
        plot2 = plt.subplot(211)
        plot2.set_title("Original Data vs Distorted Data")
        ln1 = plot2.plot(x_pos[:,0], x_pos[:,1], 'b', label="position", alpha=0.5)
        # ax2 = plot2.twinx()
        ln3 = plot2.plot(acceleration[:,0], acceleration[:,1], 'r', label="distorted accelerometer data")
        lines = ln1+ln3
        labels = [l.get_label() for l in lines]
        plot2.legend(lines, labels, loc=2)

        ## Plot fixed data.
        plt.figure(1)
        plot1 = plt.subplot(212)
        plot1.set_title("Original Data vs Fixed Data")
        ln1 = plot1.plot(x_pos[:,0], x_pos[:,1], 'b', label="position", alpha=0.5)
        # ax2 = plot1.twinx()
        ln3 = plot1.plot(fixed_acceleration[:,0], fixed_acceleration[:,1], 'r', label="fixed accelerometer data")
        lines = ln1+ln3
        labels = [l.get_label() for l in lines]
        plot1.legend(lines, labels, loc=2)

        plot1.set_xlim([0, acceleration[-1,0]])
        plot2.set_xlim([0, acceleration[-1,0]])
        plot3.set_xlim([0, acceleration[-1,0]])

        plt.show()
