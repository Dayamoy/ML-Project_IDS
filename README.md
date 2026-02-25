# Machine Learning-Based Intrusion Detection System

This project implements a Machine Learning-based Intrusion Detection System (IDS) using the **CICIDS 2017** dataset. The IDS employs **Support Vector Machine (SVM)** and **Naive Bayes** classifiers to detect network intrusions via binary classification (Benign vs. Attack), achieving accuracies of **91.10%** and **64.26%**, respectively.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Acknowledgments](#acknowledgments)

## Introduction

Intrusion Detection Systems are crucial for identifying unauthorized access or anomalies in network traffic. This project leverages machine learning techniques to build an IDS capable of distinguishing between normal (benign) and malicious activities across multiple attack categories.

## Dataset

The [CICIDS 2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html) is utilized for training and evaluating the models. This dataset includes benign and up-to-date common attacks, closely resembling real-world data. It provides comprehensive network traffic data with labeled flows based on timestamp, source and destination IPs, ports, protocols, and attack types.

The merged dataset comprises **188,939 samples** with **78 features** and **1 label column**, sampled from 8 original CSV files covering multiple days of network traffic. The label distribution is:

| Label | Samples |
|---|---|
| BENIGN | 105,387 |
| DoS Hulk | 21,255 |
| PortScan | 15,896 |
| DDoS | 12,702 |
| DoS GoldenEye | 9,484 |
| FTP-Patator | 7,938 |
| SSH-Patator | 5,897 |
| DoS slowloris | 5,306 |
| DoS Slowhttptest | 5,074 |

For model training, labels are converted to binary: **BENIGN \u2192 0**, **All attacks \u2192 1**.

## Data Preprocessing

The dataset is preprocessed using the following steps:

1. **Dataset Merging**: Eight individual CSV files from CICIDS 2017 are sampled and merged into a single dataset.
2. **Data Cleaning**: Filtering out rows with negative values in flow-related features (e.g., Flow IAT Mean, Bwd IAT Std) and handling missing values via imputation.
3. **Binary Label Encoding**: Multi-class attack labels are mapped to a single attack class (1), with benign traffic as class (0).

## Feature Engineering

The dataset includes 78 features extracted using CICFlowMeter. Key feature selection steps:

- **SVM Model**: `SelectPercentile(f_classif, percentile=9)` \u2014 selects the top 9% of features (7 features):
  `Bwd Packet Length Max`, `Bwd Packet Length Min`, `Bwd Packet Length Mean`, `Bwd Packet Length Std`, `Fwd Packets/s`, `Max Packet Length`, `Average Packet Size`

- **Naive Bayes Model**: `SelectPercentile(f_classif, percentile=10)` \u2014 selects the top 10% of features (8 features):
  `Bwd Packet Length Max`, `Bwd Packet Length Min`, `Bwd Packet Length Mean`, `Bwd Packet Length Std`, `Fwd Packets/s`, `Bwd Packets/s`, `Max Packet Length`, `Average Packet Size`

## Models

Two machine learning models were implemented with an 80/20 train-test split (`random_state=303`):

- **Support Vector Machine (SVM)** with RBF kernel: Achieved an accuracy of **91.10%**.
  - Training samples: 151,151 | Test samples: 37,788
- **Multinomial Naive Bayes**: Achieved an accuracy of **64.26%**.
  - Training samples: 151,147 | Test samples: 37,787

## Results

| Model | Accuracy | Train Samples | Test Samples | Features Selected |
|---|---|---|---|---|
| **SVM (RBF Kernel)** | **91.10%** | 151,151 | 37,788 | 7 |
| **Multinomial Naive Bayes** | **64.26%** | 151,147 | 37,787 | 8 |

> **Note:** An initial exploratory notebook (`IDS.ipynb`) using only the Friday DDoS subset (225,745 samples) with Naive Bayes and 40% feature selection achieved ~80.17% accuracy, indicating that model performance varies significantly based on dataset composition and feature selection strategy.

## Usage

To replicate the results:

1. Clone the repository:
   ```bash
   git clone https://github.com/Dayamoy/ML-Project_IDS.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ML-Project_IDS
   ```
3. Install the required dependencies (`pandas`, `numpy`, `scikit-learn`).
4. Download the [CICIDS 2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html) (MachineLearningCSV format) and place the CSV files in a `MachineLearningCSV/` directory.
5. Run `merge all dataset into one.ipynb` to preprocess and merge the dataset into `main_dataset.csv`.
6. Run `IDS using SVM.ipynb` or `IDS using Naive Bayes.ipynb` to train and evaluate the models.

## Directory Structure
```
ML-Project_IDS/
\u2502
\u251c\u2500\u2500 merge all dataset into one.ipynb   # Dataset merging and preprocessing
\u251c\u2500\u2500 IDS.ipynb                          # Initial exploration (single-file, Naive Bayes)
\u251c\u2500\u2500 IDS using SVM.ipynb                # SVM classifier training and evaluation
\u251c\u2500\u2500 IDS using Naive Bayes.ipynb        # Naive Bayes classifier training and evaluation
\u251c\u2500\u2500 README.md                          # Project documentation
\u2514\u2500\u2500 _config.yml                        # GitHub Pages configuration
```

## Acknowledgments

- Dataset: [Canadian Institute for Cybersecurity \u2014 CICIDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- Feature extraction: CICFlowMeter
