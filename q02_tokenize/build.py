# %load q02_tokenize/build.py
# Default imports

from nltk.tokenize import TreebankWordTokenizer

import pandas as pd

from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data

# Write your solution here:
path = 'data/20news-bydate-train/'

def q02_tokenize(data):
    
    'write your solution here'
    twenty_train, X_train, X_test, y_train, y_test = q01_load_data(path)
    tokenizer = TreebankWordTokenizer()
    to_tokenize = pd.Series(X_train)
    almost_tokenized = to_tokenize.apply(lambda row: row.lower())
    tokenized = almost_tokenized.apply(lambda row: tokenizer.tokenize(str(row)))
    # X_train.head(20)
    return tokenized
q02_tokenize(path)

