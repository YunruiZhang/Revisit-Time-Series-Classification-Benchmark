# Revisiting Time Series Classification Benchmarks: The Impact of Temporal Information for Classification (PAKDD 2025)

This repository contains the official implementation of **"Revisiting Time Series Classification Benchmarks: The Impact of Temporal Information for Classification"**, accepted at **PAKDD 2025**.

## Evaluating the Impact of Temporal Information

Experiments corresponding to **Section 4** of the paper, which analyze the impact of **temporal information** on classification, can be found in the **`permute_benchmark`** folder.

We use the **.ts format** from the **UCR Time Series Classification Archive**, available at:  
[UCR Archive (.ts format)](http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip).

## UCR Augmented Dataset

The results and experimental code for the proposed **UCR Augmented** dataset are located in the **`UCR_AUG`** and **`UCR_AUG_pal`** folders.

The script for generating the **UCR Augmented** dataset, along with an example of how to augment time series classification datasets, can be found in **`UCR_aug.py`**.
