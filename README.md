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

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NLTK
- Scikit-learn
- Matplotlib

To install the required Python packages, run:
```bash
pip install -r requirements.txt


#### 3.6: Installation
```markdown
## Installation

Clone this repository and navigate into the project directory:

```bash
git clone https://github.com/jasonwei20/eda_nlp.git
cd eda_nlp

import nltk
nltk.download('wordnet')



#### 3.7: Dataset
```markdown
## Dataset

The dataset used is a subset of the SST-2 (Stanford Sentiment Treebank) dataset, which is located in the `data/` directory.

#### 3.8: Usage

### Running the Experiments

To run the experiments with different fractions of the dataset, execute the script:

```python
cnn_results, rnn_results = run_experiments()
plot_results(cnn_results, rnn_results)



#### 3.9: Results
```markdown
## Results

### CNN + EDA Performance:

- The CNN model demonstrated very high accuracy across all fractions of the dataset:
  - **Accuracy with 10% of the dataset**: Nearly 100%.
  - **Accuracy with 20-100% of the dataset:** Remains around 100%, with a slight decrease at the 100% mark.

### RNN + EDA Performance:

- The RNN model showed gradual improvement as the dataset size increased:
  - **Accuracy with 10% of the dataset:** Approximately 40%.
  - **Accuracy with 100% of the dataset:** Capped at around 60%.

### Hyperparameter Tuning: Alpha and n_aug

- **Tuning Observations:**
  - The accuracy of the CNN model improves with a higher number of augmented sentences (n_aug), especially at lower alpha values.
  - **Best performance** was achieved with **Alpha = 0.1** and **n_aug = 16**, where the model consistently reached accuracy close to 100%.

### Final Model Performance

- **Final Model:** Trained with the best hyperparameters (Alpha = 0.1, n_aug = 16).
- **Validation Accuracy:** The final model achieved an accuracy of nearly 100% on the validation set.
- **IMDB Test Set Accuracy:** The model achieved an accuracy of **48.98%** on the IMDB test set, indicating challenges in generalizing to different datasets.
