import numpy as np
from nltk.corpus import stopwords
import nltk
stops = set(stopwords.words('english'))

from file_parsing import parse_file
from grammatical_tools import stops, word_split

def shuffle(list):
	np.random.shuffle(list)
	return list

#lists for storing documents
train_pos = parse_file('train_pos_reviews.txt')
train_neg = parse_file('train_neg_reviews.txt')
dev_pos = parse_file('dev_pos_reviews.txt')
dev_neg = parse_file('dev_neg_reviews.txt')

#shuffles together positive and negative reviews
train_all = shuffle(train_pos + train_neg)
dev_all = shuffle(dev_pos + dev_neg)

from sklearn.feature_extraction.text import TfidfVectorizer

def get_tf_idfs(doc_list): #calculates tf-idf scores for each doc, returns those and a list of words in all docs
	num_docs = len(doc_list)
	reviews = [doc[1] for doc in doc_list]
	
	vectorizer = TfidfVectorizer(strip_accents = 'ascii', lowercase=True, tokenizer = word_split, preprocessor=None, stop_words=None)
	#stop word removal is done by the tokenizer 
	#since this function likes to remove extra stopwords that i want to keep
	#like "n't"
	
	reviews = [doc[1] for doc in train_all]
	
	tf_idfs = vectorizer.fit_transform(reviews)
	
	word_list = vectorizer.get_feature_names_out()
	
	return word_list, tf_idfs

def calc_word_scores(docs): #calculates average ratings for all documents with a word
	word_scores = {}
	for doc in docs:
		rating, review = doc
		polarity = 1 if (rating >= 7) else -1
		for word in word_split(review):
			if(word in word_scores):
				sum_polarity, sum_rating, num_ratings = word_scores[word]
				word_scores[word] = (sum_polarity + polarity, sum_rating + rating, num_ratings + 1)
			else:
				word_scores[word] = (polarity, rating, 1)
	return word_scores

#predicts the polarity of an individual review using only average word polarities 
def predict_review_polarity(tf_idf_scores, word_scores, word_list):
	polarity_estimate = 0
	for word_idx, tf_idf in enumerate(tf_idf_scores):
		word = word_list[word_idx]
		if(word in word_scores):
			word_polarity = word_scores[word][0]
			polarity_estimate += word_polarity * tf_idf
	return polarity_estimate
	
#def predict_review_polarity_negation(review, tf_idf_scores, word_scores, word_list):

def predict_polarities(word_scores, docs, max_reviews = 3000): #uses word scores from training data to predict pos or neg sentiment for docs
	word_list, tf_idfs = get_tf_idfs(docs)
	actual_vals, pred_vals = [], []
	
	word_to_idx = {word: idx for (idx, word) in enumerate(word_list)}

	#iterates through dev reviews and uses average word ratings + tf_idf scores to calculate pos vs neg
	for doc_idx, ((rating, review), tf_idf) in enumerate(zip(docs, tf_idfs)):
		polarity = 1 if (rating >= 7) else -1
		tf_idf_arr = tf_idf.toarray()[0]
		polarity_estimate = predict_review_polarity(tf_idf_arr, word_scores, word_list)
		actual_vals.append(polarity)
		pred_vals.append(1 if (polarity_estimate > 0) else -1)
		if(doc_idx % 100 == 0): print(doc_idx)
		if(doc_idx > max_reviews): break
	return np.array(actual_vals), np.array(pred_vals)

def main():
	train_word_list, train_tf_idfs = get_tf_idfs(train_all) #get training data tf-idfs

	train_word_scores = calc_word_scores(train_all) #average word ratings and polarity
	
	word_scores = {}
	
	for word in train_word_scores:
		polarity_sum, rating_sum, count = train_word_scores[word]
		if(count > 10): #gets rid of uncommon words
			word_scores[word] = (polarity_sum/count, rating_sum/count, count)

	#writes word scores to a file
	word_scores_file = open('word_scores.txt', mode='w')
	for word in train_word_list:
		if(word not in word_scores): continue
		polarity, rating, num_ratings = word_scores[word]
		word_scores_file.write(f'{word}\t{polarity}\t{rating}\t{num_ratings}\n')
	
	actual_vals, pred_vals = predict_polarities(word_scores, dev_all)

	from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score

	acc, pre, rec = accuracy_score(actual_vals, pred_vals), precision_score(actual_vals, pred_vals), recall_score(actual_vals, pred_vals)

	print('accuracy %%%.2f' % (100*acc))
	print('precision %%%.2f' % (100*pre))
	print('recall %%%.2f' % (100*rec))
	print('f-score %%%.2f' % (100* (2/(1/pre + 1/rec))))
	
	from sklearn.metrics import confusion_matrix
	
	print(confusion_matrix(actual_vals, pred_vals))

if __name__ == '__main__':
	main()
