import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
#https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8

class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        print("fitting vectorizer")
        self.vectorizer.fit(X)
        print("done fitting vectorizer...now transforming doc")
        self.transformed = super(TfidfVectorizer, self.vectorizer).transform(X)
        print("done transforming doc")
        self.avdl = self.transformed.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        #X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = self.transformed.sum(1).A1
        print("transforming query")
        q, = self.vectorizer.transform([q])
        print("done transforming query")
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
    print(target)
    print(list(result.head(10)["title"]))
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

bm25 = BM25()
print(":loading data")
data = pd.read_pickle("processed_data_small.pkl")
data = data.filter(['id', 'title'])
df = pickle.load(open("proc_sentences.pkl",'rb'))
#fulldf = pd.read_pickle("processed_data_2.pkl")
texts = df["Combine"]
print("fitting data")
bm25.fit(texts)

def query_text(query, qt = 150, start = 1):
    print("Query is : " , query)
    
    order=bm25.transform(query, texts)
    sorted_order  = np.argsort(order)
    answers=[]
    d= data
    d["Sim"] = order[1:]
    d = d.sort_values(by=["Sim"], ascending=False)
    result=d[['id' ,'title']]
    return result
