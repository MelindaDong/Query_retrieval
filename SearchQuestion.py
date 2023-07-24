import sys
input_question = sys.argv[1]
#input_question = "How do I control my emotions?"

# import libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np
import re


# data processing
# read data and preprocessings
Data = pd.read_csv('data.tsv', sep='\t', error_bad_lines=False)
Data = Data.dropna()
# drop the duplicate rows in question2
Data = Data.drop_duplicates(subset=['question2'])

# fillter the first 100 'question1' with 'is_duplicate' == 1.0
Q1 = Data[Data['is_duplicate'] == 1.0].head(100)
# keep only the 'question1' column
Q1 = Q1['question1']

Q2= Data['question2']
# test Q2 as the first 1000 'question2'
#Q2 = Data['question2'].head(1000)
Q2 = Q2.astype(str) # make sure the type is string

#Process the review column line by line to do text preprocessing
def process_review(review):
    # remove the punctuations
    review = re.sub(r"[^\w\s]+", "", review)
    # convert the review to lower case
    review = review.lower()
    # remove the stopwords
    stop_words = set(stopwords.words('english'))
    # tokenize the words
    word_tokens = word_tokenize(review)
    filtered_review = [w for w in word_tokens if not w in stop_words]
    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_review = [lemmatizer.lemmatize(w) for w in filtered_review]
    # return the processed review
    return lemmatized_review

# process the train and test reviews
#Q1 = Q1.apply(process_review)
input_question = process_review(input_question)
Q2 = Q2.apply(process_review)


# load the pre-trained glove word embeddings
embeddings_dict = {}
with open("glove/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = vector



# calculate the unigram probability for each word in the question
# create a vocabulary
vocabulary = set()
for q in Q2:
    for w in q:
        vocabulary.add(w)

vocabulary = list(vocabulary)

corpus = []
for Q in Q2:
    for word in Q:
        corpus.append(word)

# calculate the unigram probability of a word in the corpus
def calculate_unigram_probability(word):
    word_count = corpus.count(word)
    total_words = len(corpus)
    unigram_probability = word_count / total_words
    return unigram_probability

# create a dictionary to store the unigram probability of each word
unigram_probabilities = {}
for word in vocabulary:
    unigram_probabilities[word] = calculate_unigram_probability(word)


# main function

input_question_dict = {0: input_question}
#Q1_dict = Q1.to_dict() # in case, might need to run again
Q2_dict = Q2.to_dict()

def sentence_embedding(word_embeddings = embeddings_dict, sentences = Q2_dict, a = 0.5, word_probabilities = unigram_probabilities):
    sentence_embeddings = {}
    for index, s in sentences.items():
        vs = np.zeros(50)  # Initialize sentence embedding as zero vector
        for w in s:
            try:
                a_value = a / (a + word_probabilities[w])  # Smooth inverse frequency, SIF
                vs += a_value * word_embeddings[w] * (1/len(s)) # vs += sif * word_vector
                #vs += ((word_embeddings[w] * a)/(a + word_probabilities[w]))* (1/len(s))
            except KeyError:
                continue
        sentence_embeddings[index] = vs

    sentence_list = list(sentence_embeddings.values())
    num_sentences = len(sentence_list)
    embedding_dim = sentence_list[0].shape[0]  # Assuming all embeddings have the same dimension
    X = np.zeros((embedding_dim, num_sentences))

    for i, embedding in enumerate(sentence_list):
        X[:, i] = embedding

    # Perform singular value decomposition
    u, _, _ = np.linalg.svd(X, full_matrices=False)  #full_matrices=False ensures that only the necessary number of singular vectors is returned
    u = u[:, 0]  # Extract first singular vector

    for index, s in sentences.items():
        vs = sentence_embeddings[index]
        uuT = np.outer(u, u)  # Compute the outer product of u with itself
        vs = vs - np.dot(uuT, vs)  # Subtract the product of uuT and vs from vs
        sentence_embeddings[index] = vs

    return sentence_embeddings


#Q1_dict_vec2 = sentence_embedding(embeddings_dict, Q1_dict, 0.5, unigram_probabilities)
input_question_dict_vec2 = sentence_embedding(embeddings_dict, input_question_dict, 0.5, unigram_probabilities)
Q2_dict_vec2 = sentence_embedding(embeddings_dict, Q2_dict, 0.5, unigram_probabilities)

# calculate cosine similarity between 2 sentence embeddings
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

# get the top 5 similar questions
def get_top5_similar_questions3(query):
    ranking = {}
    for q in Q2_dict_vec2:
        ranking[q] = cosine_similarity(query, Q2_dict_vec2[q])
    ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    return ranking[:2], ranking[:5]

top2, top5_dict = get_top5_similar_questions3(input_question_dict_vec2[0])

# get the top 5 similar questions
top_index = []
for i in top5_dict:
    top_index.append(i[0])
# print sentences acording to the index
for i in top_index:
    print(Data['question2'].iloc[i])
    print('\n')


