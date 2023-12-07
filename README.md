# <div align="center">Twitter Disaster Classification NLP Project</div>  
# General Info
This repository contains the code and resources used in a college project focused on classifying tweets from Twitter. The project includes data preprocessing, bigram modeling, and classification using three different feature extraction methods: Binary Vectorizer, Count Vectorizer, and TF-IDF.
# File structure
- `util/preprocessor.py` includes `Preprocessor` class that implements cleaning methods such as removing URL's, punctuations, and stopwords, tokenization and lemmatization.
- `preprocessing.ipynb` uses and show how different `Preprocessor` class methods work
- `bigram_model.ipynb` building a simple Bigram model to estimate the likelyhood probabilty of a seqeuence (tweet in our context)
- `classification_models.ipynb` implements 3 differenet feature extraction methods (Binary Vectorizer, Count Vectorizer, and TF-IDF Vectorizer), for each one of them differnet modeld were built and evaulated and finally summarized to show best model and best feature extraction method.
# Project Setup
1- Clone this repository:
```bash
git clone https://github.com/PrinceEGY/NLP-Project.git
cd NLP-Project
```
2- Set up environment:
```bash
pip install -r requirements.txt
```
3- All the notebooks are ready to use and play with.
