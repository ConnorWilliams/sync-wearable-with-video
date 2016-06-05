import numpy as np
from scipy import interpolate
import fxn_library as f
from random import uniform
from scipy import signal as signal_lib
import matplotlib.pyplot as plt

def rad(x): return x*(np.pi/180)


def constant(signal, shift):
    distortion = np.ones(signal.shape[0])*shift
    distortion[0] = 0
    new_signal = apply_distortion(signal, distortion)
    return new_signal


def linear(signal, max_drift):
    distortion = np.linspace(0, max_drift, signal.shape[0])
    new_signal = apply_distortion(signal, distortion)
    return new_signal


def periodic(signal, max_drift):
    # Create the distortion wave.
    amplitude = max_drift
    frequency = 1/((signal[-1,0]-signal[0,0])/3)
    distortion = (amplitude*np.sin( (rad(signal[:,0]-signal[0,0])*frequency*360) ))
    new_signal = apply_distortion(signal, distortion)
    return new_signal


def triangular(signal, max_drift):
    # Create the distortion wave.
    amplitude = max_drift
    period = (signal[-1,0]-signal[0,0])/3
    distortion = amplitude*signal_lib.sawtooth((signal[:,0]+period/4)/(period/(2*np.pi)), 0.5)
    new_signal = apply_distortion(signal, distortion)
    return new_signal


def apply_distortion(signal, distortion):
    start_time = signal[0,0]
    end_time = signal[-1,0]
    drifted_time = signal[:,0]+distortion
    f_i = interpolate.interp1d(drifted_time, signal[:,1], bounds_error=False)
    new_y = f_i(signal[:,0])

    new_signal = np.vstack( (signal[:,0], new_y) ).T

    # plt.plot(signal[:,0], signal[:,1])
    # plt.plot(signal[:,0], distortion)
    # plt.plot(new_signal[:,0], new_signal[:,1])
    # plt.legend(['signal', 'distortion', 'new_signal'])
    # plt.show()
    # exit()
    return new_signal
