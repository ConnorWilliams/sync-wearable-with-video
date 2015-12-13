import numpy as np
np.set_printoptions(threshold=np.nan)

top_dir = "clean_data/"
date_dir = "Rec_10_06_15_15_33_04_yangdi_1/"
acc0_dir = "ACC0.000000/"
acc1_dir = "ACC1.000000/"
dev_dir = "Dev0.000000/"

ts_info_filename = top_dir + date_dir + dev_dir + 'frameTSinfo0.000000.txt'
ts_info_names = ("imName", "idxColour", "tsColour", "idxDepth", "tsDepth", "diff", "tsRecDvc")

video_filename = top_dir + date_dir + dev_dir + 'user1.txt'
video_names = ()

acc0_filename = top_dir + date_dir + acc0_dir + 'ACC_0.000000.txt'
acc1_filename = top_dir + date_dir + acc1_dir + 'ACC_1.000000.txt'
acc_names = ("idx", "diff", "tsRecDvc", "AccX", "AccY", "AccZ", "tsInternal")

ts_info = np.genfromtxt(ts_info_filename, dtype=float, comments='#', delimiter="\t",
                  skip_header=0, skip_footer=0, converters=None, missing_values=None,
                  filling_values=None, usecols=None,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=True, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)

video = np.genfromtxt(video_filename, dtype=float, comments='#', delimiter=" ",
                  skip_header=0, skip_footer=0, converters=None, missing_values=None,
                  filling_values=None, usecols=None,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=True, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)

acc0 = np.genfromtxt(acc0_filename, dtype=float, comments='#', delimiter="\t",
                  skip_header=0, skip_footer=0, converters=None, missing_values=None,
                  filling_values=None, usecols=None,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=True, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)

acc1 = np.genfromtxt(acc1_filename, dtype=float, comments='#', delimiter="\t",
                  skip_header=0, skip_footer=0, converters=None, missing_values=None,
                  filling_values=None, usecols=None,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=True, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)
print 'Files read successfully...'

# Make ts_info and video the same size to accomodate for video running before skeleton is detected
while ts_info[0][0] < video[0][0]:
    ts_info = np.delete(ts_info,0,0)
print 'Resized ts_info to be same size as video...'

# Join ts_info and video at index
video = np.delete(video,0,1)
video = np.hstack((ts_info, video))
print 'Joined ts_info and video...'

# Calculate absolute acceleration
euclid = lambda x: np.sqrt(x[3]*x[3]+x[4]*x[4]+x[5]*x[5])
acc0 = np.vstack( (acc0.T, map(euclid, acc0)) ).T
acc1 = np.vstack( (acc1.T, map(euclid, acc1)) ).T
print 'Calculated absolute acceleration...'

euclid = lambda x: np.sqrt(x[3]*x[3]+x[4]*x[4]+x[5]*x[5])
acc0 = np.vstack( (acc0.T, map(euclid, acc0)) ).T
acc1 = np.vstack( (acc1.T, map(euclid, acc1)) ).T

print video.shape
print acc0.shape
# TODO Plot absolute acceleration against XYZ
# TODO Cross-correlation between accelerometer and skeleton data
