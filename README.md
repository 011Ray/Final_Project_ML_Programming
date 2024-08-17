# Final_Project_ML_Programming
# EDA NLP Experimentation

This project implements and evaluates the effectiveness of data augmentation techniques, particularly Easy Data Augmentation (EDA), in improving the performance of text classification models on a sentiment analysis task. The project uses Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to classify sentences into positive or negative sentiment.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Final Model](#final-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

- ## Introduction

The project focuses on:
- Applying EDA to the SST-2 dataset to enhance training data with diverse sentence structures.
- Comparing the performance of CNN and RNN models trained on the original and augmented datasets.
- Conducting hyperparameter tuning for the EDA parameters to optimize model performance.
- Evaluating the final model on the IMDB test dataset.

- ## Project Structure

eda_nlp/
│
├── code/
│ └── eda.py # Contains EDA augmentation functions
├── data/
│ └── sst2_train_500.txt # Training dataset
├── final_model.keras # Trained model file
├── README.md # Project documentation
└── requirements.txt # Python dependencies

