# Machine Learning-Based Intrusion Detection System

This project implements a Machine Learning-based Intrusion Detection System (IDS) using the CICIDS 2017 dataset. The IDS employs Support Vector Machine (SVM) and Naive Bayes classifiers to detect network intrusions, achieving accuracies of 91% and 88%, respectively.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Intrusion Detection Systems are crucial for identifying unauthorized access or anomalies in network traffic. This project leverages machine learning techniques to build an IDS capable of distinguishing between normal and malicious activities.

## Dataset

The [CICIDS 2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html) is utilized for training and evaluating the models. This dataset includes benign and up-to-date common attacks, closely resembling real-world data. It provides comprehensive network traffic data with labeled flows based on timestamps, source and destination IPs, ports, protocols, and attack types.

## Features

The dataset includes numerous features extracted using CICFlowMeter, such as:

- Flow Duration
- Total Fwd Packets
- Total Backward Packets
- Total Length of Fwd Packets
- Total Length of Bwd Packets
- Packet Length Mean
- Packet Length Std
- ...and many more.

These features are instrumental in distinguishing between normal and malicious network behaviors.

## Data Preprocessing

The dataset is preprocessed using the following steps:

1. **Data Cleaning**: Handling missing values and removing duplicates.
2. **Feature Selection**: Identifying and selecting the most relevant features.
3. **Data Normalization**: Scaling features to improve model performance.

## Models

Two machine learning models were implemented:

- **Support Vector Machine (SVM)**: Achieved an accuracy of 91%.
- **Naive Bayes**: Achieved an accuracy of 88%.

## Results

The performance of the models is as follows:

- **SVM Classifier**: 91% accuracy
- **Naive Bayes Classifier**: 88% accuracy

## Usage

To replicate the results:

1. Clone the repository:
   ```bash
   git clone https://github.com/Dayamoy/ML-Project_IDS.git
2. Navigate to the project directory:
   ```bash
   cd ML-Project_IDS
3. Install the required dependencies:
4. Download the CICIDS 2017 dataset from the official website and place it in the data/ directory.
5. Preprocess the dataset and train the models.
6. Evaluate the models using the test dataset.
   
## Directory Structure
```bash
ML-Project_IDS/
│
├── data/                   # Directory for the dataset
├── scripts/                # Python scripts for preprocessing, training, and evaluation
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│
├── models/                 # Saved models after training
├── results/                # Output results and performance metrics
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation



