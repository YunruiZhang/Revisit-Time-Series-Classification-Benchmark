# Thsi file contains functions for generating the UCR AUG benchmark from the original UCR archive

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from sktime.datasets import load_from_tsfile
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier


un_normal = ['BME',
 'Chinatown',
 'Crop',
 'EOGHorizontalSignal',
 'EOGVerticalSignal',
 'GunPointAgeSpan',
 'GunPointMaleVersusFemale',
 'GunPointOldVersusYoung',
 'HouseTwenty',
 'InsectEPGRegularTrain',
 'InsectEPGSmallTrain',
 'PigAirwayPressure',
 'PigArtPressure',
 'PigCVP',
 'PowerCons',
 'Rock',
 'SemgHandGenderCh2',
 'SemgHandMovementCh2',
 'SemgHandSubjectCh2',
 'SmoothSubspace',
 'UMD']


def add_seqpad(train_x, test_x, padding_length, std=0.01):
    train_x_aug = []
    test_x_aug = []
    
    for i in range(train_x.shape[0]):
        # for front pad
        front_pad = []
        cur_mean = train_x[i][0]
        for a in range(padding_length):
            cur_mean = np.random.normal(cur_mean, std, 1)
            front_pad.append(cur_mean)
        front_pad = np.array(front_pad).flatten()
        front_pad = np.flip(front_pad)
        # for end pad
        end_pad = []
        cur_mean = train_x[i][-1]
        for b in range(padding_length):
            cur_mean = np.random.normal(cur_mean, std, 1)
            end_pad.append(cur_mean)
            
        end_pad = np.array(end_pad).flatten()
        train_x_aug.append(np.concatenate([front_pad, train_x[i], end_pad]))
    
    
    for i in range(test_x.shape[0]):
        # for end pad
        front_pad = []
        cur_mean = test_x[i][0]
        for a in range(padding_length):
            cur_mean = np.random.normal(cur_mean, std, 1)
            front_pad.append(cur_mean)
        front_pad = np.array(front_pad).flatten()
        front_pad = np.flip(front_pad)
        # for end pad
        end_pad = []
        cur_mean = test_x[i][-1]
        for b in range(padding_length):
            cur_mean = np.random.normal(cur_mean, std, 1)
            end_pad.append(cur_mean)
            
        end_pad = np.array(end_pad).flatten()
        test_x_aug.append(np.concatenate([front_pad, test_x[i], end_pad]))
        
        
    train_x_aug = np.array(train_x_aug)
    test_x_aug = np.array(test_x_aug)
    return train_x_aug, test_x_aug

def shift_ts(train_x_aug, test_x_aug, padding):
    padd = padding
    np.random.seed(0)
    
    train_x_aug_shifted = []
    test_x_aug_shifted = []

    for a in range(train_x_aug.shape[0]):
        random_shift_left = np.random.randint(1, padd)  # Include max_value in the range
        random_shift_right = padd-random_shift_left
        train_x_aug_shifted.append(train_x_aug[a][padd-random_shift_left:-(padd-random_shift_right)])

    for a in range(test_x_aug.shape[0]):
        random_shift_left = np.random.randint(1, padd)  # Include max_value in the range
        random_shift_right = padd-random_shift_left
        test_x_aug_shifted.append(test_x_aug[a][padd-random_shift_left:-(padd-random_shift_right)])

    train_x_aug_shifted=np.array(train_x_aug_shifted)
    test_x_aug_shifted=np.array(test_x_aug_shifted)
    
    return train_x_aug_shifted, test_x_aug_shifted



def normalize(x):
    for a in range(x.shape[0]):
        x[a] = (x[a]-x[a].mean())/x[a].std()


# example here we use the .ts file format from sktime 
dataset = "Car"
train_x, train_y = load_from_tsfile(f"/mnt/raid1/yunrui/data/Univariate_ts/{dataset}/{dataset}_TRAIN.ts", return_data_type="numpy2d")
test_x, test_y = load_from_tsfile(f"/mnt/raid1/yunrui/data/Univariate_ts/{dataset}/{dataset}_TEST.ts", return_data_type="numpy2d")

# there exist a set of datasets in the UCR archive that is not z-normalized 
if dataset in un_normal:
    normalize(train_x)
    normalize(test_x)

# classfication on original UCR dataset
clf = KNeighborsTimeSeriesClassifier(n_jobs=10)
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
acc_pre = accuracy_score(test_y, y_pred)

# padding length to add on each end of the time series instance
padding_percentage=0.2
padding_length = int(train_x.shape[1]*padding_percentage)

# add the padding
train_x_aug, test_x_aug = add_seqpad(train_x, test_x, padding_length)
# perform the misalignment
train_x_aug_shifted, test_x_aug_shifted = shift_ts(train_x_aug, test_x_aug, padding_length)

# classfication on the AUG UCR datset
clf_aug = KNeighborsTimeSeriesClassifier(n_jobs=10)
clf_aug.fit(train_x_aug_shifted, train_y)
y_pred_aug = clf_aug.predict(test_x_aug_shifted)
acc_aug = accuracy_score(test_y, y_pred_aug)


print(f"accuracy for dataset {dataset} before augmentation: {acc_pre} After augmentation {acc_aug}")