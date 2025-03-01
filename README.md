# Sentiment Analysis of Movie Reviews

## Overview
This project implements a comprehensive sentiment analysis model to classify IMDB movie reviews as positive or negative, leveraging advanced machine learning techniques for a university-level study. It uses the IMDB Movie Reviews dataset (50,000 reviews, split into 25,000 training and 25,000 testing reviews) and explores three models:

- **Logistic Regression:** A baseline model using TF-IDF features, achieving 87% accuracy.
- **Long Short-Term Memory (LSTM):** An advanced deep learning model with word embeddings, reaching 88% accuracy.
- **Ensemble (Random Forest + Gradient Boosting):** A robust ensemble model combining TF-IDF and metadata features (review length, sentiment score), achieving 90% accuracy with 89% cross-validation.

The project demonstrates NLP preprocessing, feature engineering, and model evaluation, supported by open-source tools like Python 3.11.1, TensorFlow, scikit-learn, NLTK, matplotlib, and textblob, run on a MacBook Air M2.

## Directory Structure

Sentiment_Analysis_Project/
├── data/
│   ├── raw/aclImdb/  # Raw IMDB dataset (excluded from Git)
│   └── processed/    # Processed data (if generated)
├── src/
│   ├── notebook.ipynb              # Initial exploratory notebook
│   ├── exploration_preprocessing.ipynb  # Data exploration and preprocessing
│   ├── modeling.ipynb             # Logistic Regression implementation
│   ├── advanced_analysis.ipynb    # LSTM deep learning model
│   └── complex_operations.ipynb   # Ensemble model (RF + GB)
├── output/
│   └── figures/
│       ├── confusion_matrix.png         # Logistic Regression confusion matrix
│       ├── confusion_matrix_lstm.png    # LSTM confusion matrix
│       └── confusion_matrix_ensemble.png  # Ensemble confusion matrix
├── docs/
│   ├── report.tex    # LaTeX report source
│   ├── report.pdf    # Compiled PDF report (excluded from Git, generated in Overleaf)
│   └── presentation.pptx  # PowerPoint presentation (excluded from Git if large)
├── requirements.txt  # Python dependencies for reproducibility
└── README.md         # This file



## Installation
To set up and run this project:

1. Clone the repository:
   ```bash
   git clone https://github.com/rashaddd777/sentiment_analysis_project.git
   cd sentiment_analysis_project
