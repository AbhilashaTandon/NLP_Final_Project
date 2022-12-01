import numpy as np
from nltk.corpus import stopwords
import nltk
stops = set(stopwords.words('english'))

from file_parsing import parse_file, rating_review_split
from grammatical_tools import stops, word_split

def shuffle(list):
	np.random.shuffle(list)
	return np.array(list)

all_vocabulary = open("aclImdb_v1\\aclImdb\\imdb.vocab", encoding = 'utf-8').read().splitlines()

#consider limiting the vocabulary to the first n terms, since these are the most common n terms

#lists for storing documents
train_pos = parse_file('train_pos_reviews.txt')
train_neg = parse_file('train_neg_reviews.txt')
dev_pos = parse_file('dev_pos_reviews.txt')
dev_neg = parse_file('dev_neg_reviews.txt')

print(len(train_pos), len(train_neg), len(dev_pos), len(dev_neg))

#shuffles together positive and negative reviews
train_all = shuffle(train_pos + train_neg)
dev_all = shuffle(dev_pos + dev_neg)

train_ratings, train_reviews = rating_review_split(train_all)
dev_ratings, dev_reviews = rating_review_split(dev_all)

#print(train_ratings)

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy import stats

normed_train_ratings = stats.zscore(train_ratings, axis=None)
normed_dev_ratings = stats.zscore(dev_ratings, axis=None)

def get_tf_idfs(doc_list): #calculates tf-idf scores for each doc, returns those and a list of words in all docs
	num_docs = len(doc_list)
	reviews = [doc[1] for doc in doc_list]
	
	vectorizer = TfidfVectorizer(strip_accents = 'ascii', lowercase=True, tokenizer = word_split, preprocessor=None, stop_words=None, vocabulary = all_vocabulary)
	#stop word removal is done by the tokenizer 
	#since this function likes to remove extra stopwords that i want to keep
	#like "n't"
	
	reviews = [doc[1] for doc in doc_list]
	
	tf_idfs = vectorizer.fit_transform(reviews)
	
	return tf_idfs
	
from sklearn.decomposition import LatentDirichletAllocation

def lda(tf_idfs, topics = 5): #does latent dirichlet analysis on tf-idf scores to find topics
	lda = LatentDirichletAllocation(n_components=topics, random_state=0)
	doc_topics = lda.fit_transform(tf_idfs)
	return lda.components_, doc_topics

from sklearn.linear_model import LinearRegression

def find_polarity_topics(doc_topics, review_polarities):
	reg = LinearRegression().fit(doc_topics, review_polarities)
	print(reg.score(doc_topics,review_polarities))
	return reg.coef_, reg.intercept_

def main():
	train_tf_idfs = get_tf_idfs(train_all) #get training data tf-idfs
	
	dev_tf_idfs = get_tf_idfs(dev_all)
	
	topic_weights, doc_topics = lda(train_tf_idfs)
	
	print(topic_weights.shape)
	
	print(doc_topics.shape)
	
	#print(doc_topics)
	
	print(find_polarity_topics(doc_topics, normed_train_ratings))
	
	#for (rating, review), topics in zip(train_all, doc_topics):
		#print(rating, review, topics)
	

if __name__ == '__main__':
	main()