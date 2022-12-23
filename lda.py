import numpy as np
import nltk

from file_parsing import parse_file, rating_review_split
from grammatical_tools import stops, word_split, negation_analyzer

def shuffle(list):
	np.random.shuffle(list)
	return np.array(list)

all_vocabulary = open("aclImdb_v1\\aclImdb\\imdb.vocab", encoding = 'utf-8').read().splitlines()

#all_vocabulary = all_vocabulary[:50000]

#consider limiting the vocabulary to the first n terms, since these are the most common n terms

#lists for storing documents
train_pos = parse_file('train_pos_reviews.txt')
train_neg = parse_file('train_neg_reviews.txt')
dev_pos = parse_file('test_pos_reviews.txt')
dev_neg = parse_file('test_neg_reviews.txt')

#dev_pos = parse_file('dev_pos_reviews.txt')
#dev_neg = parse_file('dev_neg_reviews.txt')


print(len(train_pos), len(train_neg), len(dev_pos), len(dev_neg))

#shuffles together positive and negative reviews
train_all = shuffle(train_pos + train_neg)
dev_all = shuffle(dev_pos + dev_neg)

num_topics = 1000
num_train_reviews = len(train_all)
num_dev_reviews = len(dev_all)
num_words = len(all_vocabulary)

train_ratings, train_reviews = rating_review_split(train_all)
dev_ratings, dev_reviews = rating_review_split(dev_all)

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy import stats

normed_train_ratings = stats.zscore(train_ratings, axis=None)
normed_dev_ratings = stats.zscore(dev_ratings, axis=None)

train_polarities = [2*int(rating >= 7)-1 for rating in train_ratings]
dev_polarities = [2*int(rating >= 7)-1 for rating in dev_ratings]
#uses z-scores to normalize ratings, (rating - mean)/stdev

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

def lda(tf_idfs, topics, test_tf_idfs): #does latent dirichlet analysis on tf-idf scores to find topics
	lda = LatentDirichletAllocation(n_components=topics, random_state=0)
	doc_topics = lda.fit_transform(tf_idfs)
	test_topics = lda.transform(test_tf_idfs)
	return lda.components_, doc_topics, test_topics

from sklearn.linear_model import LinearRegression

def find_polarity_topics(doc_topics, review_polarities):
	#linear regression with array of topic scores (X) and polarities (Y)
	reg = LinearRegression().fit(doc_topics, review_polarities)
	print(reg.score(doc_topics,review_polarities))
	return reg.coef_, reg.intercept_


#def lda_transform(topic_weights, tf_idfs, texts): #uses topic weights from previous lda to find topics from tf-idfs
#	return np.dot(topic_weights, tf_idfs)
#	
#	negation_masks = np.array(negation_analyzer(texts)).reshape(tf_idfs.shape)
#	return np.dot(topic_weights, np.multiply(tf_idfs, negation_masks))
#	

def predict_polarity(coeffs, intercept, doc_topics): #uses topic weights to predict polarity from topics
	return np.dot(coeffs, doc_topics) + intercept

def main():
	train_tf_idfs = get_tf_idfs(train_all) #get training data tf-idfs
	
	dev_tf_idfs = get_tf_idfs(dev_all) #get development data tf-idfs
	
	
	topic_weights, doc_topics, test_doc_topics = lda(train_tf_idfs, num_topics, dev_tf_idfs)
	#do lda on tf-idfs and apply the results to get topics for dev set
	
	# = lda_transform(topic_weights, dev_tf_idfs, dev_reviews)
	
	coefs, y_intercept = find_polarity_topics(doc_topics, train_polarities)
	#do linear regression to predict polarity from topics
	
	predicted_polarities = [predict_polarity(coefs, y_intercept, topics) > 0 for topics in test_doc_topics]
	actual_polarities = [rating >= 0 for rating in dev_polarities]
	#convert sliding scale to discrete
	
	from sklearn.metrics import precision_score, accuracy_score, recall_score
	
	acc, pre, rec = accuracy_score(actual_polarities, predicted_polarities), precision_score(actual_polarities, predicted_polarities), recall_score(actual_polarities, predicted_polarities)
	
	from sklearn.metrics import classification_report
	
	print(classification_report(actual_polarities, predicted_polarities))

if __name__ == '__main__':
	main()