import sys
import re
import nltk
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


def load_data(database_filepath):
    '''
    input:  filepath from sqlite
    output: X:message series, Y:category dataframe, category_names: the list of category    
    
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages',engine)
    genre_counts = df.groupby('genre').count()['message']
    print(genre_counts)
    genre_names = list(genre_counts.index)
    print(genre_names)
   # get column message as X variable 
    X = df['message']
   # get the columns from 5 to 40 as y
    Y = df.iloc[:,4:39]  
    category_names=Y.columns
    return X,Y,category_names


def tokenize(text):
    '''
    input: text to be cleaned
    output: cleaned text
    '''

    #normalize case  
    text= re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    

    #punctuation
    stop_words = stopwords.words("english")    

    #tokenize text
    tokens=word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return(tokens)                          

    


def build_model():
    '''
    function: build pipeline
    output: pipeline model
    
    '''
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100)))
    ])  
       
    parameters = {'clf__estimator__n_estimators':[50, 100]}
    cv = GridSearchCV(estimator=pipeline, param_grid = parameters, cv = 5,n_jobs=1)#local machine, n_jobs set to 1
   
    return cv                       
                           
                           
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input: the trained model, X test dataset, Y test dataset, category names
    output: f1 score, precision and recall
    
    '''
   # print f1 score, precision, recall(打印微调后的模型的精确度、准确率和召回率)
    y_pred = model.predict(X_test)
    
    # build classification report on every column
    performances = []
    for i in range(Y_test.shape[1]):
        performances.append([f1_score(Y_test.iloc[:, i], y_pred[:, i], average='micro'),
                             precision_score(Y_test.iloc[:, i], y_pred[:, i], average='micro'),
                             recall_score(Y_test.iloc[:, i], y_pred[:, i], average='micro')])
    # build dataframe
    performances = pd.DataFrame(performances, columns=['f1 score', 'precision', 'recall'],
                                index = category_names)   
    return performances
    


def save_model(model, model_filepath):
    '''
    input: trained model, and the path of the model
    output: pkl format model
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))                       
    pass


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
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()