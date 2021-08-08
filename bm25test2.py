import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import time
#https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8
start = time.time()
class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """

        self.vectorizer.fit(X)

        self.transformed = super(TfidfVectorizer, self.vectorizer).transform(X)

        self.avdl = self.transformed.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        #X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = self.transformed.sum(1).A1
  
        q, = self.vectorizer.transform([q])
      
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = self.transformed.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1


import pandas as pd
import pickle as pickle
def preproc(query):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    #query = 'In C++, can you define a variable in terms of other variables that have already been defined?'
    query = tokenizer.tokenize(query)
    query = [i for i in query if not i in stop_words]

    query = [lemmatizer.lemmatize(word) for word in query]
    query = [i.lower() for i in query]
    query = ' '.join([text for text in query])
    return query
def evaluate(result, target, top1 ,top3 , top5 ,top10 ,top30):

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

print("Starting setup")
bm25 = BM25()
print("Loading data")
data = pd.read_pickle("processed_data_small.pkl")
data = data.filter(['id', 'title'])
df = pickle.load(open("proc_sentences.pkl",'rb'))
#fulldf = pd.read_pickle("processed_data_2.pkl")
texts = df["Combine"]
print("fitting data")
bm25.fit(texts)
print("Setup Complete")

def query_text(query, qt = 150, start = 1):
    #print("Query is : " , query)
    
    order=bm25.transform(query, texts)
    sorted_order  = np.argsort(order)
    answers=[]
    d= data
    d["Sim"] = order[1:]
    d = d.sort_values(by=["Sim"], ascending=False)
    result=d[['id' ,'title']]
    return result
def start_eval():#call this function to run all the evaluation
    print("Starting BM25 evaluation....")

    testcases = pd.read_csv("test.csv")
    top1,top3,top5,top10 ,top30=0,0,0,0,0
    start_test = time.time()
    for i , j in testcases.iterrows():
        testcases = pd.read_csv("test.csv")
        query = j["QUERY"]
        d = query_text(query)

        result = d[['title']]
        top1,top3,top5,top10 ,top30=evaluate(result, j["TITLE"] , top1,top3,top5,top10,top30)


        #print(i, "\nThe top30 accuracy is " , top30 ,"\nThe top10 accuracy is " , top10 , " \nThe top5 accuracy is " , top5, " \nThe top3 accuracy is " , top3, "\nThe top1 accuracy is " , top1)
    end_test = time.time()
    print("BM25 EVALUATION RESULTS")
    print("There are a total of " , i, " search queries to test")
    print("\nThe top30 accuracy is " , top30 ,"\nThe top10 accuracy is " , top10 , " \nThe top5 accuracy is " , top5, " \nThe top3 accuracy is " , top3, "\nThe top1 accuracy is " , top1)
    print("Total querying time is " , end_test-start_test, " seconds")
    print("BM25 EVALUATION SUCCESSFUL")
end = time.time()
print("SET TIME FOR BM25 is " , end-start , " seconds")
# start_eval()