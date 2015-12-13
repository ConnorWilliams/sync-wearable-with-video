# Synchronising Wearable Accelerometer and Video Data

My Individual Project in 3rd Year Computer Science at The University Of Bristol

## Data Files
#### frameTSinfoXXX.txt
During the recording you will have one (or more if the recording is very long) log file, containing the time stamp information. The Name of the file is frameTSinfoXXX.txt

1. The index of the images as they are saved on the disk (from 0 to N image)
* The index of the image given by the internal frame index system  of the camera (for the COLOUR image)
* Timestamp (in microseconds) given by the internal clock of the camera (for the COLOUR image)
* The index of the image given by the internal frame index system  of the camera (for the DEPTH image)
* Timestamp (in microseconds) given by the internal clock of the camera (for the DEPTH image)
    * Timestamps for depth and color image can differ a little bit (few ms), but there is hardware synchronization that ensure they are correspondent.
* The difference in ms between the time instant in which the frames have been received by the Laptop and the received previous frame. 
* Hour information (up to ms) of the time instant in which the frames have been received by the recording machine.
    * I (Massimo Camplani) encoded on a unique integer, as we are using matlab and we need to have an ascii file with only numbers (not chars). For example 11924379 --> 1(hour).19(minutes).24(second).379(ms) . Anyway, you really don't need to parse the other elements since you can obtain them summing the sixth column.

#### ACC_XXX.txt
Information from the wearable accelerometer. The subject wears 2 accelerometers:
* On the right hand - ACC_0XXX*.txt
* On the torso/ trunk - ACC_1XXX.txt


1. Read #
2. Difference in ms between the current timestamp vs previous.
3. Timestamp of the read when the laptop received the reading.
4. X acceleration.
5. Y acceleration.
6. Z acceleration.
7. Internal time stamp of the accelerometer device.

#### userXXX.txt
Information about the user joints 3D position

1. Corresponding number of frame (the same index that is used for the image name)
* The following columns are in sets of 4 and contain the information of the joints. The first one is the confidence of the joint tracker (0 if the joint is interpolated) and the other three are the joints 3D position XYZ. The order of the joints in each rows is:
    1. JOINT_HEAD
    * JOINT_NECK
    * JOINT_LEFT_SHOULDER
    * JOINT_RIGHT_SHOULDER
    * JOINT_LEFT_ELBOW 
    * JOINT_RIGHT_ELBOW
    * JOINT_LEFT_HAND
    * JOINT_RIGHT_HAND
    * JOINT_TORSO
    * JOINT_LEFT_HIP
    * JOINT_RIGHT_HIP
    * JOINT_LEFT_KNEE
    * JOINT_RIGHT_KNEE
    * JOINT_LEFT_FOOT
    * JOINT_RIGHT_FOOT
