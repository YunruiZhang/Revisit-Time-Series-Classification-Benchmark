#!/bin/bash

# Define the list of equalLengthProblems
equalLengthProblems=(
    "ACSF1"
    "Adiac"
    "ArrowHead"
    "Beef"
    "BeetleFly"
    "BirdChicken"
    "BME"
    "Car"
    "CBF"
    "Chinatown"
    "ChlorineConcentration"
    "CinCECGTorso"
    "Coffee"
    "Computers"
    "CricketX"
    "CricketY"
    "CricketZ"
    "Crop"
    "DiatomSizeReduction"
    "DistalPhalanxOutlineAgeGroup"
    "DistalPhalanxOutlineCorrect"
    "DistalPhalanxTW"
    "Earthquakes"
    "ECG200"
    "ECG5000"
    "ECGFiveDays"
    "ElectricDevices"
    "EOGHorizontalSignal"
    "EOGVerticalSignal"
    "EthanolLevel"
    "FaceAll"
    "FaceFour"
    "FacesUCR"
    "FiftyWords"
    "Fish"
    "FordA"
    "FordB"
    "FreezerRegularTrain"
    "FreezerSmallTrain"
    "GunPoint"
    "GunPointAgeSpan"
    "GunPointMaleVersusFemale"
    "GunPointOldVersusYoung"
    "Ham"
    "Haptics"
    "Herring"
    "HouseTwenty"
    "InlineSkate"
    "InsectEPGRegularTrain"
    "InsectEPGSmallTrain"
    "InsectWingbeatSound"
    "ItalyPowerDemand"
    "LargeKitchenAppliances"
    "Lightning2"
    "Lightning7"
    "Mallat"
    "Meat"
    "MedicalImages"
    "MiddlePhalanxOutlineAgeGroup"
    "MiddlePhalanxOutlineCorrect"
    "MiddlePhalanxTW"
    "MixedShapesRegularTrain"
    "MixedShapesSmallTrain"
    "MoteStrain"
    "OliveOil"
    "OSULeaf"
    "PhalangesOutlinesCorrect"
    "Phoneme"
    "PigAirwayPressure"
    "PigArtPressure"
    "PigCVP"
    "Plane"
    "PowerCons"
    "ProximalPhalanxOutlineAgeGroup"
    "ProximalPhalanxOutlineCorrect"
    "ProximalPhalanxTW"
    "RefrigerationDevices"
    "Rock"
    "ScreenType"
    "SemgHandGenderCh2"
    "SemgHandMovementCh2"
    "SemgHandSubjectCh2"
    "ShapeletSim"
    "ShapesAll"
    "SmallKitchenAppliances"
    "SmoothSubspace"
    "SonyAIBORobotSurface1"
    "SonyAIBORobotSurface2"
    "Strawberry"
    "SwedishLeaf"
    "Symbols"
    "SyntheticControl"
    "ToeSegmentation1"
    "ToeSegmentation2"
    "Trace"
    "TwoLeadECG"
    "TwoPatterns"
    "UMD"
    "UWaveGestureLibraryX"
    "UWaveGestureLibraryY"
    "UWaveGestureLibraryZ"
    "Wafer"
    "Wine"
    "WordSynonyms"
    "Worms"
    "WormsTwoClass"
    "Yoga"
)

# Maximum number of datasets to run together
max_processes=20

# Counter for running processes
running_processes=0
padding_values=(0.1 0.3 0.4 0.5)

# Outer loop to go through each padding value
for padding in "${padding_values[@]}"; do
    echo "Starting scripts with padding = $padding"
    # Loop through each problem and execute STC_seq.py with the problem name in the background
    for problem in "${equalLengthProblems[@]}"; do
        # Check if the maximum number of processes is reached
        if ((running_processes >= max_processes)); then
            # Wait for one of the background processes to finish
            wait -n
            # Decrement the running process counter
            ((running_processes--))
        fi
        
        # Execute STC_seq.py with the problem name in the background
        python rot_seq.py --dataset "$problem" --padding $padding &
        
        # Increment the running process counter
        ((running_processes++))
    done
done
# Wait for all remaining background processes to finish
wait

echo "All scripts have finished execution for all padding values."