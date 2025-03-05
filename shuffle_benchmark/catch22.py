#import multiprocessing
import pickle
import pandas as pd
import numpy as np
#from numba import get_num_threads, njit, prange, set_num_threads

# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.linear_model import RidgeClassifierCV
# from sklearn.metrics import ConfusionMatrixDisplay

from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_from_tsfile
from sktime.classification.feature_based import Catch22Classifier
#import matplotlib.pyplot as plt
from sklearn.utils import check_random_state

import random

import warnings
# from pandas.core.common import SettingWithCopyWarning
# from pandas.errors import PerformanceWarning

import os
import time
import re
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
# warnings.simplefilter(action="ignore", category=PerformanceWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

# from functions.gpu_mul_fea import gpu_transform
# from functions.classifier import classfier

random.seed(0)
np.random.seed(0)

def normalize(x):
    for a in range(x.shape[0]):
        x[a] = (x[a]-x[a].mean())/x.std()



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
"Crop",
"DiatomSizeReduction", # 16,306,345,4
"DistalPhalanxOutlineAgeGroup", # 400,139,80,3
"DistalPhalanxOutlineCorrect", # 600,276,80,2
"DistalPhalanxTW", # 400,139,80,6
"Earthquakes", # 322,139,512,2
"ECG200",   #100, 100, 96
"ECG5000",  #4500, 500,140
"ECGFiveDays", # 23,861,136,2
"ElectricDevices", # 8926,7711,96,7
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
"HandOutlines", # 1000,370,2709,2
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
"NonInvasiveFetalECGThorax1", # 1800,1965,750,42
"NonInvasiveFetalECGThorax2", # 1800,1965,750,42
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
"StarLightCurves", # 1000,8236,1024,3
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
"UWaveGestureLibraryAll", # 896,3582,945,8
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

def stratified_resample_data(X_train, y_train, X_test, y_test, random_state=None):
    """Stratified resample data without replacement using a random state.

    Reproducible resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray 
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_X : np.ndarray or list of np.ndarray
        New train data.
    train_y : np.ndarray
        New train labels.
    test_X : np.ndarray or list of np.ndarray
        New test data.
    test_y : np.ndarray
        New test labels.
    """
    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # ensure same classes exist in both train and test
    assert list(unique_train) == list(unique_test)

    if is_array:
        shape = list(X_train.shape)
        shape[0] = 0

    X_train = np.zeros(shape) if is_array else []
    y_train = np.zeros(0)
    X_test = np.zeros(shape) if is_array else []
    y_test = np.zeros(0)

    # for each class
    for label_index in range(len(unique_train)):
        # get the indices of all instances with this class label and shuffle them
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        rng.shuffle(indices)

        train_cases = counts_train[label_index]
        train_indices = indices[:train_cases]
        test_indices = indices[train_cases:]

        # extract data from corresponding indices
        train_cases = (
            all_data[train_indices]
            if is_array
            else [all_data[i] for i in train_indices]
        )
        train_labels = all_labels[train_indices]
        test_cases = (
            all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
        )
        test_labels = all_labels[test_indices]

        # concat onto current data from previous loop iterations
        X_train = (
            np.concatenate([X_train, train_cases], axis=0)
            if is_array
            else X_train + train_cases
        )
        y_train = np.concatenate([y_train, train_labels], axis=None)
        X_test = (
            np.concatenate([X_test, test_cases], axis=0)
            if is_array
            else X_test + test_cases
        )
        y_test = np.concatenate([y_test, test_labels], axis=None)

    return X_train, y_train, X_test, y_test


try:
    sd = int(input("Enter seed: "))
    print("You entered:", sd)
except ValueError:
    print("Invalid input. Please enter a valid integer.")

class logger:
    def __init__(self, algo_list, datasets):
        self.df = pd.DataFrame(columns=algo_list, index=datasets)
        self.n_algo = len(algo_list)
        self.n_dataset = len(datasets)
        
    def log(self, dataset, results):
        # check if the dataset is in the 
        self.df.loc[dataset] = results
        self.df.to_csv(f"c22_seed{sd}.csv")



# rck_res = pd.read_csv("./../utsc_rocket.csv")
# rck_res = rck_res.loc[rck_res['accuracy_mean']<0.9]
# rck_res.reset_index(inplace=True, drop=True)
# datasets = rck_res.dataset.values

datasets = equalLengthProblems

my_log = logger(["c22", "c22_shu"], datasets)

directory = './../../data/Univariate_ts'
subfolders = [ f.path for f in os.scandir(directory) if f.is_dir() ]




for a in subfolders:
    x = re.search("([^/]+$)", a)
    dataset_name = x.group(1)
    real_res = []
    if dataset_name in equalLengthProblems:
        train_x, train_y = load_from_tsfile(f"{a}/{dataset_name}_TRAIN.ts", return_data_type="numpy2d")
        test_x, test_y = load_from_tsfile(f"{a}/{dataset_name}_TEST.ts", return_data_type="numpy2d")
        if dataset_name in un_normal:
            normalize(train_x)
            normalize(test_x)
        if sd != 0:
            train_x, train_y, test_x, test_y = stratified_resample_data(train_x, train_y, test_x, test_y, random_state=sd)
        print(f"----------------------------------------------------")
        print(f"dataset : {dataset_name}")
        print(f"shape of train {train_x.shape}")
        print(f"shape of test {test_x.shape}")
        
        clf = Catch22Classifier(n_jobs=10, random_state=sd)
        clf.fit(train_x, train_y)
        res = clf.predict(test_x)
        cr = classification_report(test_y, res, output_dict = True)
        real_res.append(cr['accuracy'])
        print(f"------------------------catch22 acc is {cr['accuracy']}--------------------------------------")

        # do the shuffle
        np.random.seed(sd)
        tub = np.arange(train_x.shape[1])
        np.random.shuffle(tub)

        train_x = train_x[:, tub]
        test_x = test_x[:, tub]
    #################################################
        clf = Catch22Classifier(n_jobs=10,  random_state=sd)
        clf.fit(train_x, train_y)
        res = clf.predict(test_x)
        cr = classification_report(test_y, res, output_dict = True)
        real_res.append(cr['accuracy'])
        print(f"------------------------catch22 shuffle acc is {cr['accuracy']}--------------------------------------")



        my_log.log(dataset_name, real_res)
       
        