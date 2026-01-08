# NLP-BookClassificationModel
Book Genre Detection (Fiction vs Non-Fiction) using NLP

This project implements a Natural Language Processing (NLP) pipeline to automatically classify books as Fiction or Non-Fiction based on textual data (book titles/descriptions).
It was developed as part of the IT9002 – Natural Language Processing course.

Project Overview

With the rapid growth of digital libraries and online bookstores, manually categorizing books by genre has become inefficient. This project addresses the problem by using machine learning and NLP techniques to automatically detect whether a book belongs to the Fiction or Non-Fiction category.

The system performs:

Text preprocessing

Feature extraction using TF-IDF and Bag of Words

Genre classification using machine learning models

Performance evaluation using multiple metrics

Models Used

Two supervised machine learning models were implemented and compared:

Multinomial Naïve Bayes

Used as a baseline model

Fast and effective for text-based problems

Logistic Regression (Balanced)

Uses class_weight="balanced" to handle class imbalance

Achieved the best overall performance

Selected as the final model

Technologies & Libraries

Python

NLTK

Scikit-learn

Pandas

NumPy

Matplotlib

NLP Pipeline
1. Text Preprocessing

Lowercasing

Regex-based noise removal

Tokenization

Lemmatization

Stopword removal

2. Text Representation

Bag of Words (CountVectorizer)

N-grams (Unigrams + Bigrams)

TF-IDF Vectorization

POS Tagging (for linguistic analysis)

3. Classification

Multinomial Naïve Bayes

Logistic Regression (balanced)

4. Evaluation

Accuracy

Precision

Recall

F1-score (Macro & Weighted)

Confusion Matrix

Balanced Accuracy

ROC-AUC

Project uses two datasets.
1- Hugging face dataset https://huggingface.co/datasets/TheBritishLibrary/blbooksgenre
2- Open library dataset (code given in files)

Install the following packages for this project:
pip install nltk scikit-learn pandas numpy matplotlib
import nltk
nltk.download('punkt')
nltk.download('wordnet')




Author
Fatema Sadri.

