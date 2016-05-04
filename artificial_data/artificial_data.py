import matplotlib.pyplot as plt
import numpy as np
import fxn_library as f
import temporal_distortion

start_time = 0
end_time = 100
sampling_rate = 50

x_pos, acceleration = f.generateData(False, end_time, 200, 0.5, sampling_rate)

x_pos = f.getVel_Acc(x_pos)
# acceleration = temporal_distortion.constant(acceleration, 0.5)
acceleration = temporal_distortion.linear(acceleration, 5, sampling_rate)
# acceleration = temporal_distortion.periodic(acceleration, -1, 1, sampling_rate)
# acceleration = temporal_distortion.triangular(acceleration, -1, 1, sampling_rate)

cross_corr, norm_cross_corr = f.x_corr(x_pos[:,3], acceleration[:,1])

_, plotnum = plt.subplots(3, sharex=True)

winSize = 1
stepSize = 0.5

# correct_times is a function of the incorrect drifted times.
# correct_times = f(incorrect times)
f = f.sliding_xcorr(x_pos, acceleration, winSize, stepSize, plotnum[1])

new_times = f(acceleration[:,0])

plotnum[0].set_title("Original Data with triangular distortion.")
plotnum[0].plot(x_pos[:,0], x_pos[:,3])
plotnum[0].plot(acceleration[:,0], acceleration[:,1])
plotnum[0].legend(['calculated acceleration', 'accelerometer'])

string = "Sliding X-Corr with window size %1.1fs and step size %1.1f" % (winSize, stepSize)
plotnum[1].set_title(string)

plotnum[2].set_title("Fixed Data")
plotnum[2].plot(x_pos[:,0], x_pos[:,3])
plotnum[2].plot(new_times, acceleration[:,1])
plotnum[2].legend(['calculated acceleration', 'accelerometer'])
plt.xlabel("time")
plt.show()
