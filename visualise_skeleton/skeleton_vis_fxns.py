import sys, joints
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reads in a data file and returns it as a numpy ND-array.
def readFile(fileName, delimeter):
    return np.genfromtxt(fileName, dtype=float, comments='#', delimiter=delimeter,
                      skip_header=0, skip_footer=0, converters=None, missing_values=None,
                      filling_values=None, usecols=None,
                      names=None, excludelist=None, deletechars=None, replace_space='_',
                      autostrip=True, case_sensitive=True, defaultfmt='f%i',
                      unpack=None, usemask=False, loose=True, invalid_raise=True)

# Extracts part of the data file
def extract_skeleton(whole, start_time, end_time, time_idx):
    partial = whole[whole[:,time_idx]>start_time, :]
    partial = partial[partial[:,time_idx]<end_time, :]
    return partial

# Turns the data file in to lots of .png in /out directory.
def makeImage(data):
    # Each element in data is:
        # Index of images as saved on disk.
        # Index of colour images as internal frame index system of camera
        # Timestamp (in microseconds) of internal clock of camera
        # Index of depth images as internal frame index system of camera
        # Timestamp (in microseconds) of internal clock of camera
        # Difference in ms between laptop receiving frames.
        # Hour information (up to ms) when frames received by camera.
        # JOINT_HEAD CONF, X_POS, Y_POS, Z_POS
        # JOINT_NECK CONF, X_POS, Y_POS, Z_POS
        # JOINT_LEFT_SHOULDER CONF, X_POS, Y_POS, Z_POS
        # JOINT_RIGHT_SHOULDER CONF, X_POS, Y_POS, Z_POS
        # JOINT_LEFT_ELBOW CONF, X_POS, Y_POS, Z_POS
        # JOINT_RIGHT_ELBOW CONF, X_POS, Y_POS, Z_POS
        # JOINT_LEFT_HAND CONF, X_POS, Y_POS, Z_POS
        # JOINT_RIGHT_HAND CONF, X_POS, Y_POS, Z_POS
        # JOINT_TORSO CONF, X_POS, Y_POS, Z_POS
        # JOINT_LEFT_HIP CONF, X_POS, Y_POS, Z_POS
        # JOINT_RIGHT_HIP CONF, X_POS, Y_POS, Z_POS
        # JOINT_LEFT_KNEE CONF, X_POS, Y_POS, Z_POS
        # JOINT_RIGHT_KNEE CONF, X_POS, Y_POS, Z_POS
        # JOINT_LEFT_FOOT CONF, X_POS, Y_POS, Z_POS
        # JOINT_RIGHT_FOOT CONF, X_POS, Y_POS, Z_POS

    minx = data[:, 8::4].min()
    minz = data[:, 9::4].min()
    miny = data[:, 10::4].min()
    maxx = data[:, 8::4].max()
    maxz = data[:, 9::4].max()
    maxy = data[:, 10::4].max()

    fig = plt.figure(figsize=(4,6))
    ax_skeleton = fig.add_subplot(1, 1, 1, projection='3d')

    for i in range(0, len(data)):
        ax_skeleton.cla()
        drawSkeleton(data[i])

        ax_skeleton.auto_scale_xyz( [minx, maxx], [miny, maxy], [minz, maxz] )
        ax_skeleton.xaxis.set_ticklabels( [] )
        ax_skeleton.yaxis.set_ticklabels( [] )
        ax_skeleton.zaxis.set_ticklabels( [] )
        fig.tight_layout()

        plt.savefig( "out/%05d.png" %(i) )
        sys.stdout.write("\rGenerated image %i of %i" %(i+1, len(data)))
        sys.stdout.flush()

def drawSkeleton(frame):
    for n in joints.jointNeighbours:
        a = frame[7 + 4*n[0] + np.arange(1,4,1)]
        b = frame[7 + 4*n[1] + np.arange(1,4,1)]
        plt.plot([a[0],b[0]], [a[2],b[2]], [a[1],b[1]], marker="o", c="k")
