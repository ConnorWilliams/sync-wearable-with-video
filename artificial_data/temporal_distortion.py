import numpy as np
from scipy import interpolate
import fxn_library as f
from random import uniform

from scipy import signal as signal_lib
import matplotlib.pyplot as plt


def rad(x): return x*(np.pi/180)


def constant(signal, shift):
    distortion = np.ones(signal.shape[0])*shift
    signal[:,0] += distortion
    return signal


def linear(signal, max_drift, sampling_rate):
    distortion = np.linspace(0, max_drift, signal.shape[0])
    new_signal = apply_distortion(signal, distortion)
    return new_signal


def periodic(signal, min_mult, max_mult, sampling_rate):
    # Create the distortion wave.
    amplitude = (max_mult-min_mult)/2
    y_intersept = (max_mult+min_mult)/2
    frequency = 0.03
    distortion = (amplitude*np.sin( rad(signal[:,0])*frequency*360 )) + y_intersept
    new_signal = apply_distortion(signal, distortion)
    return new_signal


def triangular(signal, min_mult, max_mult, sampling_rate):
    # Create the distortion wave.
    amplitude = (max_mult-min_mult)/2
    y_intersept = (max_mult+min_mult)/2
    period = 20
    distortion = amplitude*signal_lib.sawtooth(signal[:,0]/(period/(2*np.pi)), 0.5) + y_intersept
    new_signal = apply_distortion(signal, distortion)
    return new_signal


def apply_distortion(signal, distortion):
    start_time = signal[0,0]
    end_time = signal[-1,0]
    drifted_time = signal[:,0]+distortion
    f_i = interpolate.interp1d(drifted_time, signal[:,1], bounds_error=False)
    new_y = f_i(signal[:,0])

    new_signal = np.vstack( (signal[:,0], new_y) ).T
    new_signal = f.extract_section(new_signal, start_time, end_time)

    # plt.plot(signal[:,0], signal[:,1])
    # plt.plot(signal[:,0], distortion)
    # plt.plot(new_signal[:,0], new_signal[:,1])
    # plt.legend(['signal', 'distortion', 'new_signal'])
    # plt.show()
    # exit()
    return new_signal
