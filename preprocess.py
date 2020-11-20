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
     
            center_filename = ntpath.basename(line[0])
          
            filename_angles.append([center_filename, steering_center])
            
    with open(data_out, 'w') as filehandle:
        for line in filename_angles:
            filehandle.write('%s' % line[0])
            filehandle.write(',' )
            filehandle.write('%f' % float(line[1]))
            filehandle.write('\n')


if __name__ == '__main__':
    preprocess()

    
