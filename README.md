# Vietnamese Sentiment Analysis

This repository contains code and documentation for a machine learning project that performs sentiment analysis on Vietnamese text. The project compares various approaches—including traditional machine learning models, deep learning models using token index and TF-IDF representations, and an ensemble stacking classifier—to analyze Vietnamese user comments and predict sentiment.

---

## Table of Contents

- [Problem Statement and Motivation](#problem-statement-and-motivation)
- [Dataset Overview](#dataset-overview)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architectures](#model-architectures)
  - [Token Index Deep Learning Model](#token-index-deep-learning-model)
  - [TF-IDF Deep Learning Model](#tf-idf-deep-learning-model)
  - [Machine Learning Models and Stacking Classifier](#machine-learning-models-and-stacking-classifier)
- [Experimental Setup and Training](#experimental-setup-and-training)
- [Results and Discussion](#results-and-discussion)
- [References](#references)

---

## Problem Statement and Motivation

Vietnamese social media and e-commerce platforms generate a large volume of user comments and reviews every day. There is a growing need for automated sentiment analysis to help businesses understand consumer opinions. This project is motivated by:

- **Language Challenges:** Vietnamese is a tonal language with complex word boundaries and a mix of accented/unaccented texts.
- **Business Applications:** Effective sentiment analysis can inform marketing strategies and customer service improvements.
- **Comparative Analysis:** Evaluating modern deep learning approaches versus traditional machine learning methods on a low-resource language dataset.

---

## Dataset Overview

The dataset used in this project was sourced from Kaggle and contains over 31,000 Vietnamese user comments. Key characteristics include:

- **Class Labels:** 
  - Positive (tích cực)
  - Neutral (trung lập)
  - Negative (tiêu cực)
- **Preprocessing:** Irrelevant columns and duplicate rows were removed, and sentiment labels were mapped into their Vietnamese counterparts with corresponding numerical encodings.

---

## Preprocessing Pipeline

To handle the nuances of the Vietnamese language, an advanced preprocessing pipeline was developed, consisting of:

- **Text Cleaning:** Removal of punctuation, emojis, and normalization (including lowercasing and reduction of character repetitions).
- **Tokenization:** Using tools such as [UnderTheSea](https://pypi.org/project/pyvi/) and [ViTokenizer](https://pypi.org/project/pyvi/) to accurately segment Vietnamese text.
- **Normalization:** Diacritic normalization and spacing adjustments to address challenges with accented/unaccented text.
- **Label Mapping and Encoding:** Mapping string labels to numerical values for model training.

This pipeline ensures that the text is clean, consistent, and suitable for both deep learning and machine learning models.

---

## Model Architectures

### Token Index Deep Learning Model

- **Input Representation:** Sequences of token indices generated from the preprocessed text.
- **Architecture:**
  - **Embedding Layer:** Converts token indices into dense vectors that capture semantic relationships.
  - **Convolutional Layers:** Extract local patterns (e.g., n-grams) from the text.
  - **Bidirectional LSTM with Attention:** Captures long-range dependencies and focuses on relevant parts of the text.
  - **Normalization & Dropout:** Mitigate overfitting and stabilize learning.
  - **Global Pooling and Dense Layers:** Aggregate information and output class probabilities using a final softmax layer.

### TF-IDF Deep Learning Model

- **Input Representation:** TF-IDF vectors created from the cleaned text.
- **Architecture:**
  - **Dense Layers:** Process the fixed-length TF-IDF features.
  - **Batch Normalization and Dropout:** Improve training stability and prevent overfitting.
  - **Residual Connections:** Maintain gradient flow and enable deeper network training.
  - **Final Dense Output Layer:** Uses softmax to output class probabilities.
- **Advantage:** Fixed-length input enables faster training than sequence-based models despite having a higher parameter count.

### Machine Learning Models and Stacking Classifier

- **Individual Models:** Evaluated several traditional classifiers (e.g., Naive Bayes, Logistic Regression, SVM, Random Forest, Ridge Classifier) using both TF-IDF and token index features.
- **Stacking Classifier:**
  - **Base Models:** Random Forest, AdaBoost, and SVM.
  - **Meta-Model:** Logistic Regression with balanced class weights.
  - **Rationale:** Combining multiple models leverages their individual strengths, leading to improved generalization.

---

## Experimental Setup and Training

- **Data Splitting:**
  - **Deep Learning Models:** 70% training, 15% validation, 15% testing.
  - **Machine Learning Models:** 80% training, 10% validation, 10% testing.
- **Callbacks:** Training is managed using TensorFlow callbacks:
  - **ModelCheckpoint:** Saves the best model based on validation loss.
  - **EarlyStopping:** Halts training if performance plateaus.
  - **ReduceLROnPlateau:** Adjusts the learning rate when validation loss stagnates.
- **Training Duration:** Models are trained for a fixed number of epochs with the best checkpoint restored for final evaluation.

## References 
Vietnamese Sentiment Analysis Dataset: Kaggle Dataset

Pyvi: Pyvi on PyPI

PhoBERT: PhoBERT GitHub Repository

PhoW2V: PhoW2V GitHub Repository

TensorFlow: TensorFlow Official Website

Detailed Report: Refer to the provided project report for a comprehensive discussion of methodologies and experimental results.
