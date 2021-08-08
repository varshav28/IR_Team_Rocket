import pandas as pd 
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
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

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
doc_encoding = pickle.load(open('doc_encoding_bert_full.pickle','rb'))
data = pd.read_pickle("processed_data_small.pkl")
def query_text(query, qt = 150, start = 1):
    print("Query is : " , query)
    query_encoding = model.encode(query)

    order=cosine_similarity([query_encoding] , doc_encoding)
    sorted_order  = np.argsort(-order)
    answers=[]
    d= data
    d["Sim"] = order[0]
    d = d.sort_values(by=["Sim"], ascending=False)
    result=d[['id' ,'title']]
    return result
top1,top3,top5,top10=0,0,0,0
top30 = 0
testcases = pd.read_csv("test.csv")
for i , j in testcases.iterrows():
    
    query = j["QUERY"]
    d = query_text(query)

    result = d[['title']]
    top1,top3,top5,top10 ,top30=evaluate(result, j["TITLE"] , top1,top3,top5,top10,top30)


    print(i, "\nThe top30 accuracy is " , top30 ,"\nThe top10 accuracy is " , top10 , " \nThe top5 accuracy is " , top5, " \nThe top3 accuracy is " , top3, "\nThe top1 accuracy is " , top1)


