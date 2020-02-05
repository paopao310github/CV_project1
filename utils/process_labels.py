# process labels

import numpy as np

def encode_labels(mask):
    encode_mask = np.zeros([mask.shape[0],mask.shape[1]])

    ignoreInEval_num = [213,206,207, # 1 
                        211,208, # 2 
                        216,215, # 3 
                        218,219, # 4 
                        232, # 5
                        202,231,230,228,229,233, # 7
                        212,223, # 8 
                        249,255] # 0
    background_num = [0]

    valid_labels = [
                    [200,204,209], # 1
                    [201,203], # 2
                    [217], # 3
                    [210], # 5
                    [214], # 6
                    [220,221,222,224,225,226], # 7
                    [205,227,250],# 8
                    ]

    for i in range(len(valid_labels)):
        valid_label = valid_labels[i]
        for l in valid_label:
            encode_mask[mask == l] = i+1
    
    return encode_mask

def decode_labels(labelimg):
    decode_mask = np.zeros([labelimg.shape[0],labelimg.shape[1]])

    mapping = {
        '1': 200,
        '2': 201,
        '3': 217,
        '4': 210,
        '5': 214,
        '6': 220,
        '7': 205
    }

    valid_labels = [1,2,3,4,5,6,7]
    for vl in valid_labels:
        decode_mask[labelimg == vl] = mapping[str(vl)]

    return decode_mask

