import sys
import pandas as pd
from sqlalchemy import *


def load_data(messages_filepath, categories_filepath):
    '''
    input: filepath for messages and categories
    output: merged df
    
    '''
    messages = pd.read_csv(messages_filepath)    
    categories = pd.read_csv(categories_filepath) 
    # merge datasets
    df = messages.merge(categories, left_on='id',right_on='id')
    return(df)
    pass


def clean_data(df):
    '''
    Input: merged datasets from load_data
    output: cleaned datasets which will be saved into database
    
    '''
    
    # 1. Split categories into separate category columns
    
    ## create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    ## select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(map(lambda x:x.split('-')[0],row.values.tolist()))
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # 2. Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # 3. Replace categories column in df with new category columns
    ## drop the original categories column from `df`
    df=df.drop(['categories'],axis=1)
    ## concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # 4. Remove duplicates
    ## check number of duplicates
    df.duplicated().sum()
    ## drop duplicates
    df=df.drop_duplicates()
    ## check number of duplicates
    df.duplicated().sum()
    
    return(df)    
    
    pass


def save_data(df, database_filename):
    '''
    input: tablename: df, database_file_name: database_filename
    
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False,if_exists='replace')    
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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