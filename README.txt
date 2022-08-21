Submitted by Ayush Sahni
Rollno 21111019
INFORMATION RETRIEVAL
------------------------------------------------------------------------------------

Project Dependencies:  nltk, numpy

How to install dependencies:  make install

How to retrieve documents: make run ARGV=<input_query_path>

Note: The input query file must be of type txt  
--------------------------------------------------------------------------------------

Directory structure:
    21111019-assignment1
        --- 21111019-ir-systems
                --- assgn1.ipynb
                --- retriever.py
                --- run.sh
                --- Makefile
                ---doc-dict.pkl
                ---tokens-dict.pkl
        --- 21111019-qrels
        --- README.txt

To retrieve documents, an input file must be given which contains query_id and corresponding queries in the format "Q01	How good is ketchup" ,without quotes
This file should be pasted inside folder 21111019-assignment1\21111019-ir-systems\.
Go to the folder folder 21111019-assignment1\21111019-ir-systems\ and open terminal from their incase you want to run jupyter notebook manually as absolute path is given in code.


Output files: QRels_boolean.txt, QRels_tfidf.txt, QRels_bm25.txt


In my system, it takes nearly 5 minutes to run 20 queries including some pre-processing overhead. So, let it run for sufficient time depending on the number of queries provided.

