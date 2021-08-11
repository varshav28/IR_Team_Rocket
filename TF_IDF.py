import numpy as np 
import pandas as pd 
#import os
#import re
#import seaborn as sns
import string
import nltk
import time
#from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

#from nltk.stem import WordNetLemmatizer 
#from nltk import pos_tag
#from nltk.corpus import wordnet
#import matplotlib.pyplot as plt
#from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
#import os
#import numpy as np
#import re
#import numpy as np                                  #for large and multi-dimensional arrays
#import pandas as pd                                 #for data manipulation and analysis
#import nltk                                         #Natural language processing tool-kit
import pickle
#from collections import defaultdict
#import copy
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#from nltk.corpus import stopwords                   #Stopwords corpus
#from nltk.stem import PorterStemmer                 # Stemmer
#from nltk.tokenize import word_tokenize 
#from sklearn.feature_extraction.text import TfidfVectorizer

#import re, string, unicodedata                          # Import Regex, string and unicodedata.
								   # Import contractions library.
#from bs4 import BeautifulSoup   
from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
#from nltk.tokenize import word_tokenize, sent_tokenize  # Import Tokenizer.
#from nltk.stem.wordnet import WordNetLemmatizer         # Import Lemmatizer.
#nltk.download('averaged_perceptron_tagger')
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
#import nltk
#from scipy import sparse
#from scipy import sparse as sp
start = time.time()
print("Starting setup")
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

data = pd.read_pickle("processed_data_small.pkl")
data = data.filter(['id', 'title'])
print("loaded processed data 2")
df = pd.read_pickle("proc_sentences.pkl")
print("loaded proc sentences")
tfidf = pickle.load(open("tfidf2.pickle" , "rb"))#TfidfVectorizer().fit_transform(df["Combine"])
print("loaded tfidf")
queryTFIDF_vector = pickle.load(open("querytfidf2.pickle" , "rb"))#TfidfVectorizer().fit(df["Combine"])
print("loaded query tfidf")
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
print("Setup Complete")

def query_preproc(query):
	tokenizer = RegexpTokenizer(r'\w+')
	stop_words = set(stopwords.words('english'))
	#query = 'In C++, can you define a variable in terms of other variables that have already been defined?'
	query = tokenizer.tokenize(query)
	query = [i for i in query if not i in stop_words]

	query = [lemmatizer.lemmatize(word) for word in query]
	query = [i.lower() for i in query]
	query = ' '.join([text for text in query])
	return query
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
def evaluate(result, target, top1 ,top3 , top5 ,top10 ,top30):
	#print(target)
	#print(list(result.head(10)["title"]))
	["title"]
	if target in list(result.head(30)["title"]):
		top30+=1
	if  target in list(result.head(10)["title"]):
		top10+=1
		
	if target in list(result.head(5)["title"]):
		top5+=1
	if target in list(result.head(3)["title"]):
		top3+=1
	if target in list(result.head(1)["title"]):
		top1+=1
	return top1,top3,top5,top10,top30

def start_eval():#run this to start eval
	print("Starting TFIDF evaluation....")
	testcases = pd.read_csv("test.csv")
	queryTFIDF_vector = pickle.load(open("querytfidf2.pickle" , "rb"))#TfidfVectorizer().fit(df["Combine"])
	MRR=0
	denom=0
	top1,top3,top5,top10 ,top30=0,0,0,0,0
	start_test = time.time()
	for i , j in testcases.iterrows():
		
		query = query_preproc(j["QUERY"])
		queryTFIDF = queryTFIDF_vector.transform([query])
		sim = cosine_similarity(queryTFIDF, tfidf).flatten() 
		d = data
		d['Sim'] = sim[1:]
		d = d.sort_values(by = ['Sim'], ascending = False)

		result = d[['title']]
		top1,top3,top5,top10 ,top30=evaluate(result, j["TITLE"] , top1,top3,top5,top10,top30)
		try:
			MRR+= 1/(1+list(result.head()["title"]).index(j["TITLE"]))
			
		except:
			pass

		#print(i, "\nThe top30 accuracy is " , top30 ,"\nThe top10 accuracy is " , top10 , " \nThe top5 accuracy is " , top5, " \nThe top3 accuracy is " , top3, "\nThe top1 accuracy is " , top1)
	end_test = time.time()
	print(i)
	MRR = MRR/i
	print("TFIDF EVALUATION RESULTS")
	print("There are a total of " , i, " search queries to test")
	print("\nThe top30 accuracy is " , top30 ,"\nThe top10 accuracy is " , top10 , " \nThe top5 accuracy is " , top5, " \nThe top3 accuracy is " , top3, "\nThe top1 accuracy is " , top1)
	print("Total querying time is " , end_test-start_test, " seconds")
	print("TFIDF EVALUATION SUCCESSFUL")
	print("MRR is ", MRR)

end = time.time()
print("SET TIME FOR TFIDF is " , end-start , " seconds")
# start_eval()

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