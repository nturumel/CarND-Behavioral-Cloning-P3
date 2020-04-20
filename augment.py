import os
import csv
import numpy as np
import itertools
import sys
os.chdir(sys.path[0])

def augment():
    car_images=[]
    steering_angles=[]
    samples=[]
    with open(r".\data\driving_log.csv") as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        for line in samples:
            steering_center=float(line[3])

            correction=0.2
            steering_left=str(float(steering_center)+correction)
            steering_right=str(float(steering_center)-correction)

            img_center=line[0]
            img_left=line[1]
            img_right=line[2]


            # add images and angles to data set
            car_images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_center, steering_left, steering_right])

        samples=[list(a) for a in zip(car_images, steering_angles)]


        with open('modified.csv', 'w') as filehandle:
            for listitem in samples:
                filehandle.write('%s' % listitem[0])
                filehandle.write(',' )
                filehandle.write('%s' % listitem[1])
                filehandle.write('\n')
