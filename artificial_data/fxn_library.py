import numpy as np
from scipy import interpolate
from scipy import signal
from scipy import array
import matplotlib.pyplot as plt
import sys
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import scale
import seaborn as sns
import itertools

np.set_printoptions(threshold=np.nan)


def generateData(noise, seconds, amplitude, frequency, sampling_rate):
    add_noise = noise
    time = np.arange(0, seconds, 1.0/sampling_rate)
    x_pos = amplitude * np.sin( rad(time)*frequency*360 )

    if add_noise == True:
        x_pos = x_pos + np.random.normal(0, 0.001, x_pos.size)

    acceleration = np.gradient(np.gradient(x_pos))
    x_pos = np.vstack((time, x_pos)).T
    acceleration = np.vstack((time, acceleration)).T
    return x_pos, acceleration

def rad(x): return x*(np.pi/180)


def getVel_Acc(pos_signal):
    x_vel = np.gradient(pos_signal[:,1])
    x_acc = np.gradient(x_vel)
    pos_signal = np.hstack((pos_signal, x_vel[:,None]))
    pos_signal = np.hstack((pos_signal, x_acc[:,None]))
    return pos_signal


def extract_section(whole, start_time, end_time):
    partial = whole[whole[:,0]>start_time, :]
    partial = partial[partial[:,0]<end_time, :]
    return partial


def sliding_xcorr(v, a, winSize, step, sub_plot):
    maxTime = max( max(v[:,0]), max(a[:,0]) )
    minTime = min( min(v[:,0]), min(a[:,0]) )
    sub_plot.plot()

    shift_before = 0
    shift_now = 0
    summ = 0
    count = 0

    # Do the work
    offset = winSize/2
    shifts = [0]
    times = [0]
    for windowStart in np.arange(minTime, maxTime-winSize, step):
        windowEnd = windowStart + winSize
        pv = extract_section(v, windowStart, windowEnd)
        pa = extract_section(a, windowStart+offset, windowEnd+offset)
        # windowFunc = np.hanning(len(pv))
        # cross_corr, norm_cross_corr = x_corr(pv[:,3]*windowFunc, pa[:,1]*windowFunc)
        cross_corr, norm_cross_corr = x_corr(pv[:,3], pa[:,1])
        height = cross_corr.shape[0]

        x = np.ones(height)*windowStart+(winSize/2)
        y = np.linspace(-winSize, winSize, height)
        n = cross_corr
        if y.size==x.size+1: y=np.delete(y,-1)
        if (x.size!=y.size or y.size!=n.size or x.size!=n.size):
            error = "Error: Size Mismatch in sliding_xcorr; x.size=%.0f, y.size=%.0f, n.size=%.0f" %(x.size, y.size, n.size)
            print >>sys.stderr, error
            sys.exit()

        shift_now = y[np.argmax(cross_corr)]
        offset += shift_now
        shifts.append(offset)
        times.append((windowStart+winSize/2))
        # print "Data from %.1f to %.1f is shifted by %fs" %(windowStart, windowEnd, y[np.argmax(cross_corr)])
        # if shift_now<shift_before/100: break
        summ = summ + np.abs(shift_now-shift_before)
        count = count+1
        shift_before = shift_now
        sub_plot.scatter(x, y, marker=",", lw=0, c=n, cmap="RdBu_r")

    times_hat = np.array(shifts) + times
    print np.c_[times, times_hat]
    f_i = interpolate.interp1d(times_hat, times, bounds_error=False)
    # plt.figure()
    # plt.plot(v[:,0], v[:,0])
    # plt.plot(v[:,0], f_i(v[:,0]), "-o")
    # plt.show()
    # sub_plot.plot(times, shifts)
    drift = summ/count
    # print "Drift =", drift/step, "shift/s"
    return f_i(a[:,0])

def x_corr(v, a):
    nv = (v - np.mean(v)) /  np.std(v)
    na = (a - np.mean(a)) / (np.std(a) * (len(a)-1))
    cross_corr = signal.correlate(a, v, "full")
    norm_cross_corr = signal.correlate(na, nv, "full")
    return cross_corr, norm_cross_corr
