import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sktime.datasets import load_from_tsfile

from sklearn.ensemble import RandomForestClassifier

import os
import time
import re
import random

 
from sktime.classification.kernel_based import RocketClassifier

import argparse
from aug import add_seqpad, shift_ts


parser = argparse.ArgumentParser()
parser.add_argument("--padding", required=True, type=float, help="Padding value 0-1")
args = parser.parse_args()


equalLengthProblems=[	
                                #Train Size, Test Size, Series Length, Nos Classes
"ACSF1",
"Adiac",        # 390,391,176,37
"ArrowHead",    # 36,175,251,3
"Beef",         # 30,30,470,5
"BeetleFly",    # 20,20,512,2
"BirdChicken",  # 20,20,512,2
"BME",
"Car",          # 60,60,577,4
"CBF",                      # 30,900,128,3
"Chinatown",
"ChlorineConcentration",    # 467,3840,166,3
"CinCECGTorso", # 40,1380,1639,4
"Coffee", # 28,28,286,2
"Computers", # 250,250,720,2
"CricketX", # 390,390,300,12
"CricketY", # 390,390,300,12
"CricketZ", # 390,390,300,12
"DiatomSizeReduction", # 16,306,345,4
"DistalPhalanxOutlineAgeGroup", # 400,139,80,3
"DistalPhalanxOutlineCorrect", # 600,276,80,2
"DistalPhalanxTW", # 400,139,80,6
"Earthquakes", # 322,139,512,2
"ECG200",   #100, 100, 96
"ECG5000",  #4500, 500,140
"ECGFiveDays", # 23,861,136,2
"EOGHorizontalSignal",
"EOGVerticalSignal",
"EthanolLevel",
"FaceAll", # 560,1690,131,14
"FaceFour", # 24,88,350,4
"FacesUCR", # 200,2050,131,14
"FiftyWords", # 450,455,270,50
"Fish", # 175,175,463,7
"FordA", # 3601,1320,500,2
"FordB", # 3636,810,500,2
"FreezerRegularTrain",
"FreezerSmallTrain",
#                        "Fungi", removed because only one instance per class in train. This is a query problem
"GunPoint", # 50,150,150,2
"GunPointAgeSpan",
"GunPointMaleVersusFemale",
"GunPointOldVersusYoung",                        
"Ham",      #105,109,431
"Haptics", # 155,308,1092,5
"Herring", # 64,64,512,2
"HouseTwenty",
"InlineSkate", # 100,550,1882,7
"InsectEPGRegularTrain",
"InsectEPGSmallTrain",
"InsectWingbeatSound",#1980,220,256
"ItalyPowerDemand", # 67,1029,24,2
"LargeKitchenAppliances", # 375,375,720,3
"Lightning2", # 60,61,637,2
"Lightning7", # 70,73,319,7
"Mallat", # 55,2345,1024,8
"Meat",#60,60,448
"MedicalImages", # 381,760,99,10
"MiddlePhalanxOutlineAgeGroup", # 400,154,80,3
"MiddlePhalanxOutlineCorrect", # 600,291,80,2
"MiddlePhalanxTW", # 399,154,80,6
"MixedShapesRegularTrain",
"MixedShapesSmallTrain",
"MoteStrain", # 20,1252,84,2
"OliveOil", # 30,30,570,4
"OSULeaf", # 200,242,427,6
"PhalangesOutlinesCorrect", # 1800,858,80,2
"Phoneme",#1896,214, 1024
"PigAirwayPressure",
"PigArtPressure",
"PigCVP",
"Plane", # 105,105,144,7
"PowerCons",
"ProximalPhalanxOutlineAgeGroup", # 400,205,80,3
"ProximalPhalanxOutlineCorrect", # 600,291,80,2
"ProximalPhalanxTW", # 400,205,80,6
"RefrigerationDevices", # 375,375,720,3
"Rock",
"ScreenType", # 375,375,720,3
"SemgHandGenderCh2",
"SemgHandMovementCh2",
"SemgHandSubjectCh2",
"ShapeletSim", # 20,180,500,2
"ShapesAll", # 600,600,512,60
"SmallKitchenAppliances", # 375,375,720,3
"SmoothSubspace",
"SonyAIBORobotSurface1", # 20,601,70,2
"SonyAIBORobotSurface2", # 27,953,65,2
"Strawberry",#370,613,235
"SwedishLeaf", # 500,625,128,15
"Symbols", # 25,995,398,6
"SyntheticControl", # 300,300,60,6
"ToeSegmentation1", # 40,228,277,2
"ToeSegmentation2", # 36,130,343,2
"Trace", # 100,100,275,4
"TwoLeadECG", # 23,1139,82,2
"TwoPatterns", # 1000,4000,128,4
"UMD",
"UWaveGestureLibraryX", # 896,3582,315,8
"UWaveGestureLibraryY", # 896,3582,315,8
"UWaveGestureLibraryZ", # 896,3582,315,8
"Wafer", # 1000,6164,152,2
"Wine",#54	57	234
"WordSynonyms", # 267,638,270,25
"Worms", #77, 181,900,5
"WormsTwoClass",#77, 181,900,5
"Yoga" # 300,3000,426,2
]   

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




class logger:
    def __init__(self, algo_list, datasets, log_name):
        self.df = pd.DataFrame(columns=algo_list, index=datasets)
        self.n_algo = len(algo_list)
        self.n_dataset = len(datasets)
        self.log_name = log_name
        
    def log(self, dataset, results):
        # check if the dataset is in the 
        self.df.loc[dataset] = results
        self.df.to_csv(f"./result/{self.log_name}.csv")


def normalize(x):
    for a in range(x.shape[0]):
        x[a] = (x[a]-x[a].mean())/x[a].std()


train_xs = []
train_ys = []
test_xs = []
test_ys = []
for dataset_name in equalLengthProblems:
    in_class_distance = []
    train_x, train_y = load_from_tsfile(f"/mnt/raid1/yunrui/Univariate_ts/{dataset_name}/{dataset_name}_TRAIN.ts", return_data_type="numpy2d")
    test_x, test_y = load_from_tsfile(f"/mnt/raid1/yunrui/Univariate_ts/{dataset_name}/{dataset_name}_TEST.ts", return_data_type="numpy2d")
    if dataset_name in un_normal:
        normalize(train_x)
        normalize(test_x)
    train_xs.append(train_x)
    train_ys.append(train_y)
    test_xs.append(test_x)
    test_ys.append(test_y)



padding = args.padding
my_log = logger(["Minirocket", "Minirocket_AUG"], equalLengthProblems, f"Minirocket_seq{padding}")
for dataset in range(len(equalLengthProblems)):
    
    # print(f"for dataset {equalLengthProblems[dataset]}")
    train_x = train_xs[dataset]
    train_y = train_ys[dataset]
    test_x = test_xs[dataset]
    test_y = test_ys[dataset]
    
    padding_length = int(train_x.shape[1]*padding)
    # for dataset smooth subspace
    if padding_length == 1:
        padding_length = 2
    train_x_aug, test_x_aug = add_seqpad(train_x, test_x, padding_length)
    train_x_aug_shifted, test_x_aug_shifted = shift_ts(train_x_aug, test_x_aug, padding_length)
    
    result = []
   
    # for original dataset
    random.seed(0)
    np.random.seed(0)
    clf = RocketClassifier(rocket_transform='minirocket', n_jobs = 20)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    cr_2 = classification_report(test_y, y_pred, output_dict= True)
    result.append(cr_2['accuracy'])
    

    # for augumented dataset
    random.seed(0)
    np.random.seed(0)
    clf = RocketClassifier(rocket_transform='minirocket', n_jobs = 20)
    clf.fit(train_x_aug_shifted, train_y)
    y_pred = clf.predict(test_x_aug_shifted)
    cr_1 = classification_report(test_y, y_pred, output_dict= True)
    result.append(cr_1['accuracy'])
    

    my_log.log(equalLengthProblems[dataset], result)