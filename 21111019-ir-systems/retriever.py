
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time
import string
import os
stop_words = stopwords.words('english')

nltk.download('punkt')
nltk.download('stopwords')
import pickle

#posting list dictionary : 
file=open('tokens-dict.pkl','rb')
posting_list=pickle.load(file)
file.close()


#documents dictionary:
file=open('doc-dict.pkl','rb')
documents_list=pickle.load(file)
file.close()

print("---------please wait---------------")

def preprocess(query):
    query_tokens=nltk.word_tokenize(query)
    query_tokens=[word.lower() for word in query_tokens] 
    query_tokens=[word.translate(str.maketrans('','',string.punctuation)) for word in query_tokens]
    query_tokens=[word.translate(str.maketrans('','','1234567890')) for word in query_tokens]
        
    query_tokens=[word for word in query_tokens if word!='']

    query_tokens =[word.lower() for word in query_tokens if word not in stop_words]
        

    #removal of non-ascii characters
    for i in range(0,len(query_tokens)):
        temp=query_tokens[i].encode('ascii','ignore')
        query_tokens[i]=temp.decode()

    #stemming
    ps=PorterStemmer()
    query_tokens=[ps.stem(word) for word in query_tokens]
    
    query_tokens=[word for word in query_tokens if word in posting_list]
    return query_tokens



#creating term requency dictionary
tf_dict={}


for key in documents_list:
    for word in documents_list[key]:
        if word not in tf_dict:
            tf_dict[word]={}
        else:
            if key not in tf_dict[word]:
                tf_dict[word][key]=1
            else:
                tf_dict[word][key]=tf_dict[word][key]+1

for word in tf_dict:
    for doc_id in tf_dict[word]:
        tf_dict[word][doc_id]=tf_dict[word][doc_id]/len(documents_list[doc_id])




#BOOLEAN RETRIEVAL MODEL
def boolean_retrieval_func(query):
    query_tokens=preprocess(query)
    
    temp_set1=posting_list[query_tokens[0]]
    for i in range(1,len(query_tokens)):
        temp_set2=posting_list[query_tokens[i]]
        temp_set1=temp_set1 & temp_set2
    
    top_10=[]
    for doc_id in temp_set1:
        score=0
        for word in query_tokens:
            if doc_id not in tf_dict[word]:
                continue
            score+=tf_dict[word][doc_id]
            
        top_10.append((score,doc_id))
        
    top_10.sort(reverse=True)
    res=[]
    for i in range(min(10,len(top_10))):
        res.append(top_10[i][1])
    return res




import math
#idf_dictionary
idf_dict={}
n_docs=8635

for word in posting_list:
    n_docs_curr_word=len(posting_list[word])
    idf_dict[word]=math.log(n_docs/n_docs_curr_word)
    

vocab_dict={}    #dictionary to store words and their correspondihng index
i=0
for word in posting_list:
    vocab_dict[word]=i
    i=i+1

    
#average length of documents which is required in BM25
L=0
for doc in documents_list:
    L=L+len(documents_list[doc])
    #print(len(doc))
    
L=L/len(documents_list)    



doc_unique = {}
for doc in documents_list : 
    doc_unique[doc] = set()
    for word in documents_list[doc] :
        doc_unique[doc].add(word)





import numpy as np
vocab_len=len(posting_list)
A=np.zeros((vocab_len))



def tfidf(query):
    cosine_similarity=[]
    query_tokens= preprocess(query)
    B=np.zeros((vocab_len))
    tf_query={}
    for word in query_tokens:
        tf_query[word]=0
    for word in query_tokens:
        tf_query[word]+=1
    for word in query_tokens:
        tf_query[word]/=len(query_tokens)
        
    for word in query_tokens:
        word_index=vocab_dict[word]
        B[word_index]=tf_query[word]*idf_dict[word]
        
    normB=np.linalg.norm(B)
    documents_imp = set()
    for word in query_tokens :
        for doc_id in posting_list[word]:
            documents_imp.add(doc_id)
    #print(len(documents_imp))
    
    for doc_id in documents_imp:
        temp = np.zeros((vocab_len))
        for word in doc_unique[doc_id]:
            if doc_id not in tf_dict[word]:
                continue
            tf_value=tf_dict[word][doc_id]
            word_index=vocab_dict[word]
            temp[word_index]=idf_dict[word]*tf_value
        normTemp=np.linalg.norm(temp)
        cosine_score=np.dot(temp,B)/(normTemp*normB)
        cosine_similarity.append((cosine_score,doc_id))
        
    cosine_similarity.sort(reverse=True)
    ans=[]
    n=len(cosine_similarity)
    for i in range(min(10,len(cosine_similarity))):
        ans.append(cosine_similarity[i][1])

    return ans






def BM25(query,k=2,b=0.75):
    query_tokens=preprocess(query)
    result=[]
    N=len(documents_list)
    documents_imp = set()
    for word in query_tokens :
        for doc_id in posting_list[word]:
            documents_imp.add(doc_id)
    for doc_id in documents_imp:
        doc_score=0
        for word in query_tokens:
            if doc_id not in tf_dict[word]:
                continue
            x_idf=(N - len(posting_list[word]) + 0.5)/(len(posting_list[word])+0.5)
            x_idf=math.log(x_idf)
            
            second_term=(tf_dict[word][doc_id]*len(documents_list[doc_id]) * (k+1) )/(tf_dict[word][doc_id]*len(documents_list[doc_id]) + k*(1-b+b*len(documents_list[doc_id])/L))
            prod=x_idf*second_term
            doc_score+=prod
            
        result.append((doc_score,doc_id))
        
    result.sort(reverse=True)
    ans=[]
    for i in range(min(10,len(result))):
        ans.append(result[i][1])
        
    return ans



print("-----------document retrieval started-----------------------")

import sys

file_path = sys.argv[1]
file_read=open(file_path)
a=file_read.readlines()
file_write=open('qrels_boolean.txt','a')
for line in a:
    qid=re.sub('\t.*','',line)
    qid=re.sub('\n','',qid)
    stringh=re.sub('Q[0-9][0-9]\t','',line)
    docs_boolean=boolean_retrieval_func(stringh)
    for doc_id in docs_boolean:
        file_write.write(qid+', 1, '+doc_id+', 1\n')
    
file_read.close()
file_write.close()

file_read=open(file_path)
a=file_read.readlines()
file_write=open('qrels_tfidf.txt','a')
for line in a:
    qid=re.sub('\t.*','',line)
    qid=re.sub('\n','',qid)
    stringh=re.sub('Q[0-9][0-9]\t','',line)
    docs_tfidf=tfidf(stringh)
    for doc_id in docs_tfidf:
        file_write.write(qid+', 1, '+doc_id+', 1\n')
    
file_read.close()
file_write.close()

file_read=open(file_path)
a=file_read.readlines()
file_write=open('qrels_bm25.txt','a')
for line in a:
    qid=re.sub('\t.*','',line)
    qid=re.sub('\n','',qid)
    stringh=re.sub('Q[0-9][0-9]\t','',line)
    docs_bm25=BM25(stringh)
    for doc_id in docs_bm25:
        file_write.write(qid+', 1, '+doc_id+', 1\n')
    
file_read.close()
file_write.close()