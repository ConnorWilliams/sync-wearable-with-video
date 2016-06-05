import matplotlib.pyplot as plt
import numpy as np
import fxn_library as f
import temporal_distortion
import wave
import sys
from scipy import interpolate

start_time = 0
end_time = 100
sampling_rate = 50
x_pos, acceleration = f.generateData(0.00001, end_time, 200, 0.2, sampling_rate)

spf = wave.open('../wavs/rp-1.wav','r')

#Extract Raw Audio from Wav File
x_pos = spf.readframes(-1)
x_pos = np.fromstring(x_pos, 'Int16')
fs = spf.getframerate()
time = np.linspace(0, len(x_pos)/fs, num=len(x_pos))*1000
x_pos = np.vstack((time, x_pos)).T

original_acceleration = np.copy(x_pos)
acceleration = np.copy(original_acceleration)
print acceleration.shape

## Distort the data
acceleration = temporal_distortion.linear(acceleration, 500)
# acceleration = temporal_distortion.periodic(acceleration, int(round(period/5, -2)))
# acceleration = temporal_distortion.triangular(acceleration, 500))
# acceleration = temporal_distortion.constant(acceleration, period)

winSize=500
stepSize=10000
## correct_times is a function of the incorrect drifted times.
## correct_times = f(incorrect times)
plot3 = plt.subplot(312)
plot3.set_title("Sliding X-Corr with window size %1.1f and step size %1.1f" % (winSize, stepSize))
f_t = f.sliding_xcorr(x_pos, acceleration, winSize, stepSize, plot3)


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
original_acceleration =  original_acceleration[np.logical_not(np.isnan(fixed_acceleration[:,1]))]
acceleration =  acceleration[np.logical_not(np.isnan(fixed_acceleration[:,1]))]
new_times = new_times[np.logical_not(np.isnan(fixed_acceleration[:,1]))]
fixed_acceleration =  fixed_acceleration[np.logical_not(np.isnan(fixed_acceleration[:,1]))]


## Plot the distorted data
plt.figure(1)
plot2 = plt.subplot(211)
plot2.set_title("Original Data vs Distorted Data")
ln1 = plot2.plot(x_pos[:,0], x_pos[:,1], 'b', label="position", alpha=0.2)
ax2 = plot2.twinx()
ln2 = plot2.plot(original_acceleration[:,0], original_acceleration[:,1], 'g', label="original accelerometer data")
ln3 = plot2.plot(acceleration[:,0], acceleration[:,1], 'r', label="distorted accelerometer data", alpha=0.5)
lines = ln2+ln3
labels = [l.get_label() for l in lines]
plot2.legend(lines, labels, loc=2)

## Plot fixed data.
plt.figure(1)
plt.figure(1).suptitle("Data set %s from %s to %s using the %s. MSE(ground truth, drifted)=%f, MSE(ground truth, fixed)=%f"
    %(date_dir, start_time, end_time, joints[joint_number], mse_drift, mse_fixed) )
plot1 = plt.subplot(212)
plot1.set_title("Original Data vs Fixed Data")
ln1 = plot1.plot(x_pos[:,0], x_pos[:,1], 'b', label="position", alpha=0.2)
ax2 = plot1.twinx()
ln2 = plot1.plot(original_acceleration[:,0], original_acceleration[:,1], 'g', label="original accelerometer data")
ln3 = plot1.plot(fixed_acceleration[:,0], fixed_acceleration[:,1], 'r', label="fixed accelerometer data")
lines = ln1+ln2+ln3
labels = [l.get_label() for l in lines]
plot1.legend(lines, labels, loc=2)

plot1.set_xlim([-100, acceleration[-1,0]])
plot2.set_xlim([-100, acceleration[-1,0]])
plot3.set_xlim([-100, acceleration[-1,0]])

if plot=="True": plt.show()
