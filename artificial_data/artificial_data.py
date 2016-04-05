import matplotlib.pyplot as plt
import numpy as np
import fxn_library as f

start_time = 0
end_time = 50

x_pos, acceleration = f.generateData(False, end_time, 1, 0.5, 50)

x_pos = f.getVel_Acc(x_pos)
acceleration = f.addDrift(acceleration, 0.001, start_time, end_time)

cross_corr, norm_cross_corr = f.x_corr(x_pos[:,3], acceleration[:,1])

_, plotnum = plt.subplots(3, sharex=True)

winSize = 1
stepSize = 0.5
drift = f.sliding_xcorr(x_pos, acceleration, winSize, stepSize, plotnum[1])

new_times = acceleration[:,0] - (acceleration[:,0]*drift)

plotnum[0].set_title("Original Data")
plotnum[0].plot(x_pos[:,0], x_pos[:,3])
plotnum[0].plot(acceleration[:,0], acceleration[:,1])
plotnum[0].legend(['calculated acceleration', 'accelerometer'])
string = "Sliding X-Corr with window size %1.1fs and step size %1.1f" % (winSize, stepSize)
plt.title(string)
plotnum[2].set_title("Fixed Data")
plotnum[2].plot(x_pos[:,0], x_pos[:,3])
plotnum[2].plot(new_times, acceleration[:,1])
plotnum[2].legend(['calculated acceleration', 'accelerometer'])
plt.xlabel("time")
plt.show()
