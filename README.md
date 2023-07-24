# Query_retrieval

The aim of this project is to question retravel: given a question, it will be matching through the question library where store all the existing questions and return the most similar questions.

3 different methods has been implemented, TF-IDF from scratch, sentence embedding based on averaging the word embedding and sentence embedding based on the thesis [< A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS >](https://openreview.net/pdf?id=SyK00v5xx) 

The best result is given by sentence embedding based on the paper, huge data size(363192 samples) hold back the final result a bit.
It gives final top2 accuracy is 0.61 and the final top5 accuracy is 0.71.

 `SearchQuestion.py` only integrate the best method (sentence embedding based on the paper) 
 it can be called as >> Python SearchQuestion.py “your question”
 __make sure you have glove file downloaded to run successfully__

`question_retrieval.ipynb` described more detailed processing and experiments.
