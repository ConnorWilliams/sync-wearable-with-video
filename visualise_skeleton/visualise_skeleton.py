import sys, os
import skeleton_vis_fxns as f
import numpy as np

#---- VARIABLES (Change) ----#
time_interval = 10000 #(ms)
size_of_kernel = 15

#---- DEFINITIONS (Do not change)----#
directory = sys.argv[1]
maxTime=0
minTime=0
ts_info_filename = directory + '/frameTSinfo0.000000.txt'
ts_info_names = ("imName", "idxColour", "tsColour", "idxDepth", "tsDepth", "diff", "tsRecDvc")
video_filename = directory + '/user0.txt'
video_names = ()


#---- START COMPUTATION ----#
# Read in the files specified above in to numpy arrays.
ts_info = f.readFile(ts_info_filename, "\t")
video = f.readFile(video_filename, " ")

minTime = ts_info[0,6]
maxTime = ts_info[-1,6]
try: start_time=int(sys.argv[2])
except:
    string = "Please give a start time between %i and %i: " %(minTime, maxTime)
    start_time = int(raw_input(string))
end_time = min(start_time+time_interval, maxTime)

# Trim ts_info to accomodate for video running before skeleton is detected
while ts_info[0][0] < video[0][0]:
    ts_info = np.delete(ts_info,0,0)

# Join ts_info and video at index
video = np.delete(video,0,1)
video = np.hstack((ts_info, video))

skeleton = f.extract_skeleton(video, start_time, end_time, 6)

folder = 'out'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception, e:
        print e

print "Visualising skeleton from %i for %i seconds." %(start_time, time_interval/1000)
f.makeImage(skeleton)
print "\nView images by running eog out/*"
