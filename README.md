This is our information retrieval project for 50.045 course. We have built a stack overflow search engine using Python.


Install all the requirements from the requirements.txt file.

After that, download some of the files needed for running from the google drive link
https://drive.google.com/drive/folders/1_JjjjdKSQykjI0b3llp0UElm-Yy_ziEv?usp=sharing

To run it as a website, run app.py and go to localhost:5000. 
To run it standalone, run either bm25test2.py, TF_IDF.py or bertIR.py.

in the import statements on app.py, you can specify which method you want to use by just changing a line specifying the method 
eg: from bertIR import query_text
 or from bm25test2 import query_text


Each of the above files have their own file reqirements which can be downloaded from the above google drive.
BM25: requires processed_data_small.pkl, bm25encoded.pkl (please comment out df=pickle.load(....))
TFIDF: requires processed_data_small.pkl, querytfidf2.pickle, tfidf2.pickle, proc_sentences.pkl (again feel free to comment out df = pd.read_pickle(..))
bertIR: requires processed_data_small.pkl, doc_encoding_bert_full.pickle 

To run these standalone, uncomment the start_eval() function call to test on test dataset (test.csv also in the drive)


If you run into any file not found error, it is most definitely in the drive.


bert provides the best results with a 0.21s query time and 0.71 MRR.


