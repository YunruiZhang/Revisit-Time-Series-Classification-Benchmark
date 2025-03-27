# Revisiting Time Series Classification Benchmarks: The Impact of Temporal Information for Classification (PAKDD 2025)

## Summary

This repository contains the official implementation of **"Revisiting Time Series Classification Benchmarks: The Impact of Temporal Information for Classification"**, accepted at **PAKDD 2025**.

### Abstract

Time series classification is usually regarded as a distinct task from tabular data classification due to the importance of temporal information. However, in this paper, by performing permutation tests that disrupt temporal information on the UCR time series classification archive, the most widely used benchmark for time series classification, we identify a significant proportion of datasets where temporal information has little to no impact on classification. Many of these datasets are tabular in nature or rely mainly on tabular features, leading to potentially biased evaluations of time series classifiers focused on temporal information. To address this, we propose UCR Augmented, a benchmark based on the UCR time series classification archive designed to evaluate classifiers' ability to extract and utilize temporal information. Testing classifiers from seven categories on this benchmark revealed notable shifts in performance rankings. Some previously overlooked approaches perform well, while others see their performance decline significantly when temporal information is crucial. UCR Augmented provides a more robust framework for assessing time series classifiers, ensuring fairer evaluations.

## Evaluating the Impact of Temporal Information

Experiments corresponding to **Section 4** of the paper, which analyze the impact of **temporal information** on classification, can be found in the **`permute_benchmark`** folder.

We use the **.ts format** from the **UCR Time Series Classification Archive**, available at:  
[UCR Archive (.ts format)](http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip).

## UCR Augmented Dataset

The results and experimental code for the proposed **UCR Augmented** benchmark are located in the **`UCR_AUG`** and **`UCR_AUG_pal`** folders corresponding to **Section 6** of the paper.

The script for generating the **UCR Augmented** dataset, along with an example of how to augment time series classification datasets, can be found in **`UCR_aug.py`**.

## Citation
@misc{zhang2025revisittimeseriesclassification,
      title={Revisit Time Series Classification Benchmark: The Impact of Temporal Information for Classification}, 
      author={Yunrui Zhang and Gustavo Batista and Salil S. Kanhere},
      year={2025},
      eprint={2503.20264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.20264}, 
}

## References
Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony Bagnall, Abdullah Mueen and Gustavo Batista (2015). The UCR Time Series Classification Archive. URL www.cs.ucr.edu/~eamonn/time_series_data/

Bagnall, A., Lines, J., Bostrom, A., Large, J., Keogh, E.: The great time series
classification bake off: a review and experimental evaluation of recent algorithmic
advances (May 2017). Data Mining and Knowledge Discovery pp. 606–660

Löning, M., Bagnall, A., Ganesh, S., Kazakov, V., Lines, J., & Király, F. J. (2019). sktime: A unified interface for machine learning with time series. arXiv preprint arXiv:1909.07872.
