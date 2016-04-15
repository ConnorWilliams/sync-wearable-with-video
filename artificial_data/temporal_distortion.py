import numpy as np
from scipy import interpolate
import fxn_library as f
from random import uniform

from scipy import signal as signal_lib
import matplotlib.pyplot as plt


def rad(x): return x*(np.pi/180)


def constant(signal, shift):
    start_time = signal[0,0]
    end_time = signal[-1,0]
    for l in signal: l[0] = l[0]+shift
    new_signal = f.extract_section(signal, start_time, end_time)
    return new_signal


def linear(signal, mult, sampling_rate):
    start_time = signal[0,0]
    end_time = signal[-1,0]
    for l in signal: l[0] = l[0]*mult
    f_i = interpolate.interp1d(signal[:,0], signal[:,1])
    time = np.arange(start_time, end_time, 1.0/sampling_rate)
    time = time[time<signal[-1, 0]]
    new_y = f_i(time)

    new_signal = np.vstack( (time, new_y) ).T
    new_signal = f.extract_section(new_signal, start_time, end_time)
    return new_signal


def periodic(signal, min_mult, max_mult, sampling_rate):
    # Create the distortion wave.
    amplitude = (max_mult-min_mult)/2
    y_intersect = (max_mult+min_mult)/2
    frequency = 0.03
    distortion = (amplitude*np.sin( rad(signal[:,0])*frequency*360 )) + y_intersect
    plt.plot(signal[:,0], signal[:,1])
    # plt.show()

    # Apply the distortion to the signal
    start_time = signal[0,0]
    end_time = signal[-1,0]
    # signal[:,0] = signal[:,0]*distortion
    f_i = interpolate.interp1d(signal[:,0], signal[:,1], bounds_error=False)
    new_t = signal[:,0]+distortion
    new_y = f_i(new_t)

    new_signal = np.vstack( (new_t, new_y) ).T
    new_signal = f.extract_section(new_signal, start_time, end_time)
    plt.plot(signal[:,0], new_y)
    plt.show()
    exit()
    return new_signal


def triangular(signal, min_mult, max_mult, sampling_rate):
    # Create the distortion wave.
    amplitude = (max_mult-min_mult)/2
    y_intersect = (max_mult+min_mult)/2
    period = 20
    distortion = amplitude*signal_lib.sawtooth(signal[:,0]/(period/(2*np.pi)), 0.5) + y_intersect

    # Apply the distortion to the signal
    start_time = signal[0,0]
    end_time = signal[-1,0]
    signal[:,0] = signal[:,0]*distortion
    f_i = interpolate.interp1d(signal[:,0], signal[:,1])
    time = np.arange(start_time, end_time, 1.0/sampling_rate)
    time = time[time<signal[-1, 0]]
    new_y = f_i(time)

    new_signal = np.vstack( (time, new_y) ).T
    new_signal = f.extract_section(new_signal, start_time, end_time)
    return new_signal
