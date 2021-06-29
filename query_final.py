import numpy as np 
import pandas as pd 
import os
import re
import seaborn as sns
import string
import nltk
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords
from bs4 import BeautifulSoup 
import pandas as pd
import os
import numpy as np
import re
import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import nltk                                         #Natural language processing tool-kit
import pickle
from collections import defaultdict
import copy
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer

import re, string, unicodedata                          # Import Regex, string and unicodedata.
                                   # Import contractions library.
from bs4 import BeautifulSoup   
from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
from nltk.tokenize import word_tokenize, sent_tokenize  # Import Tokenizer.
from nltk.stem.wordnet import WordNetLemmatizer         # Import Lemmatizer.
nltk.download('averaged_perceptron_tagger')
from sklearn.metrics.pairwise import linear_kernel
import nltk
from scipy import sparse
from scipy import sparse as sp
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
data = pd.read_pickle("processed_data_2.pkl")
df = pd.read_pickle("proc_sentences.pkl")
tfidf = pickle.load(open("tfidf2.pickle" , "rb"))#TfidfVectorizer().fit_transform(df["Combine"])

queryTFIDF_vector = pickle.load(open("querytfidf2.pickle" , "rb"))#TfidfVectorizer().fit(df["Combine"])
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def query_text(query):

	#variables

	#query = 'In C++, can you define a variable in terms of other variables that have already been defined?'
	query = tokenizer.tokenize(query)
	query = [i for i in query if not i in stop_words]
	print(query)
	query = [lemmatizer.lemmatize(word) for word in query]
	query = [i.lower() for i in query]
	query = ' '.join([text for text in query])
	queryTFIDF = queryTFIDF_vector.transform([query])
	sim = linear_kernel(queryTFIDF, tfidf).flatten()
	d = data
	d['Sim'] = sim[1:]
	d = d.sort_values(by = ['Sim'], ascending = False)
	result = d[['id','title']]
	return result

#variables
#data = pd.read_pickle("processed_data_2.pkl")
#df = pd.read_pickle("proc_sentences.pkl")
#tfidf = pickle.load(open("tfidf2.pickle" , "rb"))#TfidfVectorizer().fit_transform(df["Combine"])

#queryTFIDF_vector = pickle.load(open("querytfidf2.pickle" , "rb"))#TfidfVectorizer().fit(df["Combine"])
#while(True):
	#query=input("Enter query: ")
	#query = query_text(query)
	#queryTFIDF = queryTFIDF_vector.transform([query])
	#sim = linear_kernel(queryTFIDF, tfidf).flatten()
	#d = data
	#d['Sim'] = sim[1:]
	#d = d.sort_values(by = ['Sim'], ascending = False)

	#result = d[['title']]
	#print(result.head())	