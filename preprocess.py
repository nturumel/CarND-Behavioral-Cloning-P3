import os
import csv
import itertools
import sys
import ntpath


correction = 0.2
data_csv = R".\data\driving_log.csv"
data_out = R".\data\filenames_angles.csv"

def preprocess():
    filename_angles = []
    # change later
    with open(data_csv) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            steering_center = float(line[3])
            steering_left = str(float(steering_center) + correction)
            steering_right = str(float(steering_center) - correction)

            center_filename = ntpath.basename(line[0])
            left_filename = ntpath.basename(line[1])
            right_filename = ntpath.basename(line[2])

            filename_angles.append([center_filename, steering_center])
            #filename_angles.append([left_filename, steering_left])
            #filename_angles.append([right_filename, steering_right])
            
    with open(data_out, 'w') as filehandle:
        for line in filename_angles:
            filehandle.write('%s' % line[0])
            filehandle.write(',' )
            filehandle.write('%f' % float(line[1]))
            filehandle.write('\n')


if __name__ == '__main__':
    preprocess()

    
