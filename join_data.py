import numpy as np
np.set_printoptions(threshold=np.nan)

data_dir = "clean_data/"

ts_info_filename = data_dir + 'Rec_10_06_15_15_33_04_yangdi_1/Dev0.000000/frameTSinfo0.000000.txt'
ts_info_names = ("idx", "2", "3", "4", "5", "6", "7")

acc0_filename = data_dir + 'Rec_10_06_15_15_33_04_yangdi_1/ACC0.000000/ACC_0.000000.txt'
acc1_filename = data_dir + 'Rec_10_06_15_15_33_04_yangdi_1/ACC1.000000/ACC_1.000000.txt'
acc_names = ("1", "2", "3", "4", "5", "6", "7")

video_filename = data_dir + 'Rec_10_06_15_15_33_04_yangdi_1/Dev0.000000/user1.txt'
video_names = ()

ts_info = np.genfromtxt(ts_info_filename, dtype=None, comments='#', delimiter="\t",
                  skip_header=0, skip_footer=0, converters=None, missing_values=None,
                  filling_values=None, usecols=None,
                  names=ts_info_names, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=True, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)

acc0 = np.genfromtxt(acc0_filename, dtype=None, comments='#', delimiter="\t",
                  skip_header=0, skip_footer=0, converters=None, missing_values=None,
                  filling_values=None, usecols=None,
                  names=acc_names, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=True, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)

acc1 = np.genfromtxt(acc1_filename, dtype=None, comments='#', delimiter="\t",
                  skip_header=0, skip_footer=0, converters=None, missing_values=None,
                  filling_values=None, usecols=None,
                  names=acc_names, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=True, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)

video = np.genfromtxt(video_filename, dtype=None, comments='#', delimiter="\ ",
                  skip_header=0, skip_footer=0, converters=None, missing_values=None,
                  filling_values=None, usecols=None,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=True, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)
