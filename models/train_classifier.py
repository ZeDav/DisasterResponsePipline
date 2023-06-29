# train_classifier.py
# import libraries

# general
import pandas as pd
import numpy as np
import sqlalchemy
import re
import sys
from sqlalchemy import create_engine

# tokenization
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'words', 'averaged_perceptron_tagger', 'maxent_ne_chunker'])
from nltk import sent_tokenize, pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Feature Engineering
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split

# High dimensional output
from sklearn.datasets import make_multilabel_classification

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier

# ML models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import LinearSVC

#metrics
from sklearn.metrics import classification_report

#save model
import pickle

import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    Loads data from SQL Database and transforms dataframe to train model
    :param database_filepath: SQL database file (string)
    
    :returns X: Features (dataframe)
    :returns Y: Target (dataframe)
    :returns sub_category: Target labels (list)
    """
        
    # read in file and load to dataframe
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Categorized_Responses', con=engine)

    # define features matrix   
    X = df['message']
    
    #define targer/label matrix
    Y = df.iloc[:, 4:]
    #define categories
    sub_category = Y.columns

    return X, Y, sub_category


def tokenize(text):
    """
    Tokenize text data
    :param text: Messages (string)
    
    :return clean_tokens: Normalized, tokenized and lemmatized Message (list) 
    """
    #remove punctuation  
    text = re.sub(r'[^\w\s]', '', text)
    
    #tokenize
    tokens = word_tokenize(text)
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    
    # create list to return and fil with tokens
    clean_tokens = []
    for tok in tokens:
        #normalize 
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
  
    clean_tokens = [''.join(ele) for ele in clean_tokens]
    
    return clean_tokens


def build_model():
    """
    Builds a model using MultiOutputClassifier, OneVsRestClassifier and LinearSVC. 
    Data is transformed in pipeline.
    
    :return cv: Trained model after performing grid search (GridSearchCV model)
    """
    
  # define pipeline with estimators including a few transformers and a classifier
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 1), max_df=0.5, max_features=5000)),
            ('tfidf', TfidfTransformer()),
        ])),
        ('multi_clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
    
    # define parameters to perform grid search on pipeline
    parameters = {
        #'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'text_pipeline__vect__max_features': (None, 5000)
    }
    
    # create GridSearchCV object with pipeline and return as final model
    model_pipeline = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring='f1_macro', cv=2, n_jobs=1, verbose=10)
        
    return model_pipeline


def classification_report_df(report):
    """
    Function to create pandas dataframe from classification report.
    :param report: Classification report created with sklearn.metrics.classification_report
    :returns df_cl_report: Dataframe containing the classification report
    """
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-6]:
        row = {}
        row_data = line.split(' ')
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    df_cl_report = pd.DataFrame.from_dict(report_data)
    
    return df_cl_report

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Measures model's performance on test data and prints out results.
    :param model: trained model (GridSearchCV Object)
    :param X_test: Test features (dataframe)
    :param Y_test: Test targets (dataframe)
    :param sub_category: Target labels (list)
    
    :return model_performace_report: Dataframe with Model Performance report
    """
    
    # predict target values Y_pred of test features
    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names=category_names))
    
    report = classification_report(Y_test, Y_pred, target_names=category_names)
    model_performace_report = classification_report_df(report)
    
    print("\nBest Parameters:", model.best_params_)
    
    return model_performace_report


def save_model(model, model_filepath):
    """
    Save trained model as pickle file.
    :param model: Trained model (GridSearchCV Object)
    :param model_filepath: Filepath to store model (string)
    
    :return: None
    """
    # save model
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))

def save_report(model_performace_report):
    """
     Save the score of the trained model on test data
    
    :param: df_report: classification performance report dataframe
    :return: None
     """
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    
    model_performace_report.to_sql('model_performace_report', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        model_performace_report = evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        print('Saving model performance...')
        save_report(model_performace_report)

        print('Model perfomance saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()