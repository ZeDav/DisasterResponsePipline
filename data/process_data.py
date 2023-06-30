# processs_data.py

# import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(message_data_file, category_data_file):
    """
    Load data from csv file
    :param message_data_file: Messages (.csv file)
    :param category_data_file: category (.csv file)
    
    :return df: On id merged dataset as pandas dataframe 
    """
    # read in file
    messages = pd.read_csv(message_data_file)
    categories = pd.read_csv(category_data_file)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Clean data
    :param df: pandas dataframe
    
    :return df: pandas dataframe 
    """
    # split categories
    categories = df.categories.str.split(';', expand=True)
    
    # rename the columns of `categories` after categories attributes
    category_colnames = categories.iloc[0].str.replace('(-).*', '')
    categories.columns = category_colnames
    
    # only keep digits in category columns
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # ensure that datatype is int
        categories[column] = categories[column].astype('int')
    
    # Deal with invalide values: ensure only ones and 0 in the categories dataframe 
    categories = categories.replace(2, 1)
 
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column], errors='coerce')
    
    # replace column old categoreis in df with new one 
    # drop the original categories column from `df`: [[A], [B], [Categories]] --> [[A], [B]]
    df = df.drop('categories', axis=1)
    
    # add new colums to dataframe: df = [[A], [B]] --> [[A], [B], [category_colnames 1], ... [category_colnames n]]
    df = pd.concat([df, categories], axis=1)
    
    return df


def save_df(df, data_filename):
    """
    save Dataframe
    :param df: pandas dataframe
    :param data_filename: filename (str)
    """
    engine = create_engine('sqlite:///' + data_filename)
    df.to_sql('Categorized_Responses', engine, if_exists='replace', index=False) 

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, data_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(data_filename))
        save_df(df, data_filename)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

        
if __name__ == '__main__':
    main()
