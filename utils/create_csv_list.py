# create the file list and save it into a csv file

# topdir = '/root/data/LaneSeg'
#   Gray_Label
#       Label_road02
#           Label
#               Record001
#                   'Camera 5'
#                       yymmdd_*********_Camera_5_bin.png
#                   'Camera 6'
#                       yymmdd_*********_Camera_6_bin.png
#               Record002...
#               Record003...
#               Record004...
#               Record005...
#               Record006...
#               Record007...
#       Label_road03...
#       Label_road04...
#   Image_Data
#       Road02
#           Record001
#               'Camera 5'
#                   yymmdd_*********_Camera_5.jpg
#               'Camera 6'
#                   yymmdd_*********_Camera_6.jpg
#           Record002...
#           Record003...
#           Record004...
#           Record005...
#           Record006...
#           Record007...
#       Road03...
#       Road04...
#   label_fixed

import os, glob 
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def make_list(topdir):
    count_valid = 0
    count_invalid = 0

    image_list = []
    label_list = []

    imagedir = os.path.join(topdir,'Image_Data')
    print (os.path.exists(topdir))
    print (imagedir)

    for fd_road in glob.glob(imagedir + '/*/'): # Road02/03/04
        print (fd_road)
        fd_road_label = fd_road.replace('Image_Data','Gray_Label')
        fd_road_label = fd_road_label.replace('Road0','Label_road0')
        print (os.path.exists(fd_road_label))

        for fd_record in glob.glob(fd_road + '*/'): # Record001/002/003/004
            fd_record_label = fd_record.replace(fd_road, fd_road_label)
            fd_record_label = fd_record_label.replace('Record00','Label/Record00')

            for fd_camera in glob.glob(fd_record + '*/'): # camera 5/6
                fd_camera_label = fd_camera.replace(fd_record, fd_record_label)

                fnames = glob.glob(fd_camera + '*.jpg')

                for fn in fnames:
                    _,f = os.path.split(fn)
                    #print (f)
                    fn_label = os.path.join(fd_camera_label,f[:-4] + '_bin.png')
                    if os.path.exists(fn_label):
                        image_list.append(fn)
                        label_list.append(fn_label)
                        count_valid += 1
                    else:
                        count_invalid += 1
                        count_total = count_valid + count_invalid
                        print ("Warning: the corresponding label file is not found, %d out of %d cases"%(count_invalid, count_total))
                        print (fn)
                        print (fn_label)

    print ('In total %d images found'%len(image_list))
    print ('In total %d labels found'%len(label_list))

    return image_list, label_list

if __name__== '__main__':
    topdir = '/root/data/LaneSeg/'

    image_list, label_list = make_list(topdir)

    data = pd.DataFrame({'image':image_list, 'label':label_list})
    data_shuffle = shuffle(data)

    train_data = data.sample(frac=0.7, random_state=0,axis=0)
    val_data = data[~data.index.isin(train_data.index)]

    train_data.to_csv('train.csv', index=False)
    val_data.to_csv('val.csv', index=False)

