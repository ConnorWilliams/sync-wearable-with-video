import numpy as np
from scipy import interpolate
from scipy import signal
from scipy import array
from scipy import integrate
import matplotlib.pyplot as plt
import sys
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import scale
import seaborn as sns
import itertools


np.set_printoptions(threshold=np.nan)

def generateData(wgn, pn, seconds, amplitude, frequency, sampling_rate):
    time = np.arange(0, seconds, 1.0/sampling_rate)
    x_pos = amplitude * np.sin( rad(time)*frequency*360 )
    x_pos = np.vstack((time, x_pos)).T
    acceleration = np.copy(x_pos)

    x_pos[:,0] = x_pos[:,0] + np.random.normal(-pn, pn, x_pos[:,0].size)
    x_pos[:,1] = x_pos[:,1] + np.random.normal(-wgn, wgn, x_pos[:,1].size)
    # acceleration[:,0] = acceleration[:,0] + np.random.normal(-pn, pn, acceleration[:,0].size)
    acceleration[:,1] = acceleration[:,1] + np.random.normal(-wgn, wgn, acceleration[:,1].size)

    return x_pos, acceleration

def rad(x): return x*(np.pi/180)

def getVel_Acc(pos_signal):
    x_vel = np.gradient(pos_signal[:,1])
    x_acc = np.gradient(x_vel)
    pos_signal = np.hstack((pos_signal, x_vel[:,None]))
    pos_signal = np.hstack((pos_signal, x_acc[:,None]))
    return pos_signal

def getVel_Pos(acc_signal):
    times = acc_signal[:,0]
    x_acc = acc_signal[:,1]
    x_vel = integrate.cumtrapz(acc_signal[:,1], x=times, initial=0)
    x_pos = integrate.cumtrapz(x_vel, x=times, initial=0)
    acc_signal = np.vstack((times, x_pos)).T
    # acc_signal = np.vstack((acc_signal, x_vel)).T
    # acc_signal = np.vstack((acc_signal, x_acc)).T
    return acc_signal


def extract_section(whole, start_time, end_time):
    partial = whole[whole[:,0]>=start_time, :]
    partial = partial[partial[:,0]<end_time, :]
    return partial


def sliding_xcorr(v, a, windowSize, step, sub_plot):
    maxTime = max(v[-1,0], a[-1,0])
    minTime = min(v[0, 0], a[0, 0])

    window_lag = 0
    lags = [0]
    true_times = [0]
    for i, windowStart in enumerate(np.arange(minTime, maxTime-windowSize, step)):
        windowEnd = windowStart + windowSize
        pv = extract_section(v, windowStart, windowEnd)
        ## Plus the previous window lag so we get the DATA we want. It will be shifted along a bit.
        pa = extract_section(a, windowStart+window_lag, windowEnd+window_lag)
        if windowEnd+window_lag > a[-1,0]: break
        cross_corr, norm_cross_corr = x_corr(pv[:,1], pa[:,1])
        height = cross_corr.shape[0]

        x = np.ones(height)*(windowStart)                ## Start of the window
        y = np.linspace(-windowSize, windowSize, height) ## Every cross_corr value gets a point
        n = norm_cross_corr                              ## Color the points as a fxn of the cross_corr values

        if (x.size!=y.size or y.size!=n.size or x.size!=n.size):
            error = "Error: Size Mismatch in sliding_xcorr; x.size=%.0f, y.size=%.0f, n.size=%.0f" %(x.size, y.size, n.size)
            print >>sys.stderr, error
            sys.exit()

        # To make it monotonically increasing:
        # windowStart+lags[-1]+y[np.argmax(np.abs(n))] >= windowStart-step+lags[-1]
        # windowStart+y[np.argmax(np.abs(n))] >= windowStart-step
        # y[np.argmax(np.abs(n))] >= -step
        # y[np.argmax(np.abs(n))]+step >= 0
        yzz = y[y+step>0]
        nzz = n[y+step>0]
        window_lag += yzz[np.argmax(nzz)]
        lags.append(window_lag)
        true_times.append(windowStart)
        # print "Data from %.1f to %.1f is shifted by %f. \twindow_lag = %.5f" %(windowStart, windowEnd, y[np.argmax(n)], window_lag)
        sub_plot.scatter(x, y, marker=",", lw=0, c=n, cmap="RdBu_r")

    wrong_times = np.array(true_times)+lags
    f_i = interpolate.interp1d(wrong_times, true_times, bounds_error=False)
    # sub_plot.plot(true_times, lags)
    return f_i

def x_corr(v, a):
    nv = (v - np.mean(v)) /  np.std(v)
    na = (a - np.mean(a)) / (np.std(a) * (len(a)-1))
    cross_corr = signal.correlate(a, v, "full")
    norm_cross_corr = signal.correlate(na, nv, "full")
    return cross_corr, norm_cross_corr
