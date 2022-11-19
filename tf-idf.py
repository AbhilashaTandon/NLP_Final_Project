from re import sub #used for fixing line breaks
import numpy as np
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
#list of common words that we dont need to calculate average scores for 
from sklearn.feature_extraction.text import TfidfVectorizer
import profile

from nltk.tag import pos_tag

def parse_doc(doc): #parses "doc", line in file containing a review and its score, and returns those separately along with simplifying the review somewhat
	splitted = doc.split('\t')
	rating, review = splitted[0], '\t'.join(splitted[1:]) #score and review are separated by a tab but also tabs still occur in the reviews
	review = review.lower() #this may cause issues with say Bush vs bush but it probably fixes more problems than it causes
	review = review.replace('<br /><br />', '\n') #replace line break symbol with actual line break
	review = sub(r'([^A-Za-z0-9\s])',' \g<1> ', review) #makes non alphanumeric characters separate tokens
	#this also has issues since it doesnt take into account emoticons, like :-) which are great expressions of sentiment but are multiple non-alphanumeric characters
	return (rating, review)
	
def parse_file(file_name): #converts file to list of tuples of (rating, review)
	doc_list = []
	with open(file_name, mode='r', encoding='utf-8') as file:
		for line in file.readlines():
			rating, review = parse_doc(line)
			doc_list.append((int(rating), review))
	return doc_list
	
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

def get_tf_idfs(doc_list): #calculates tf-idf scores for each doc, returns those and a list of words in all docs
	num_docs = len(doc_list)
	reviews = [doc[1] for doc in doc_list]
	
	vectorizer = TfidfVectorizer(strip_accents = 'ascii', lowercase=True, preprocessor=None, stop_words=stops)
	
	reviews = [doc[1] for doc in train_all]
	
	tf_idfs = vectorizer.fit_transform(reviews)
	
	word_list = vectorizer.get_feature_names_out()
	
	return word_list, tf_idfs
	
def calc_word_scores(docs): #calculates average ratings for all documents with a word
	word_scores = {}
	
	for doc in docs:
		rating, review = doc
		pos_neg = 1 if (rating >= 7) else -1
		for word in review.split():
			if(word in stops): #stop words probably have completely neutral sentiment
				continue
			if(word in word_scores):
				sum_pos_neg, sum_rating, num_ratings = word_scores[word]
				word_scores[word] = (sum_pos_neg + pos_neg, sum_rating + rating, num_ratings + 1)
			else:
				word_scores[word] = (pos_neg, rating, 1)
	return word_scores

def predict_ratings(word_scores, docs): #uses word scores from training data to predict pos or neg sentiment for docs
	word_list, tf_idfs = get_tf_idfs(docs)
	actual_vals, pred_vals = [], []
	
	#iterates through dev reviews and uses average word ratings + tf_idf scores to calculate pos vs neg
	for doc_idx, ((rating, review), tf_idf) in enumerate(zip(docs, tf_idfs)):
		pos_neg = 1 if (rating >= 7) else -1
		tf_idf_arr = tf_idf.toarray()[0]
		pos_neg_estimate = 0
		rating_estimate = 0
		total_tf_idf = sum(tf_idf_arr)
		for word_idx, val in enumerate(tf_idf_arr):
			word = word_list[word_idx]
			if(word in word_scores):
				avg_pos_neg = word_scores[word][0]
				avg_rating = word_scores[word][1]
				rating_estimate += avg_rating * val
				pos_neg_estimate += avg_pos_neg * val
		#print(doc_idx, pos_neg, pos_neg_estimate, rating, rating_estimate/sum(tf_idf_arr))
		actual_vals.append(pos_neg)
		pred_vals.append(1 if (pos_neg_estimate > 0) else -1) #idk if this is better than 1 if 	(pos_neg_estimate > 1) else -1
		if(doc_idx % 100 == 0): print(doc_idx)
		if(doc_idx > 2000): break #limit cause the program runs slowly rn
	return actual_vals, pred_vals

def main():
	train_word_list, train_tf_idfs = get_tf_idfs(train_all) #get training data tf-idfs

	train_word_scores = calc_word_scores(train_all) #average word ratings and pos_neg
	
	word_scores = {}
	
	for word in train_word_scores:
		pos_neg_sum, rating_sum, count = train_word_scores[word]
		if(count > 100): #gets rid of uncommon words
			word_scores[word] = (pos_neg_sum/count, rating_sum/count, count)

	#writes word scores to a file
	word_scores_file = open('word_scores.txt', mode='w')
	for word in train_word_list:
		if(word not in word_scores): continue
		pos_neg, rating, num_ratings = word_scores[word]
		word_scores_file.write(f'{word}\t{pos_neg}\t{rating}\t{num_ratings}\n')
	
	
	actual_vals, pred_vals = predict_ratings(word_scores, dev_all)

	actual_vals, pred_vals = np.array(actual_vals), np.array(pred_vals)

	from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score

	print(accuracy_score(actual_vals, pred_vals))
	print(precision_score(actual_vals, pred_vals))
	print(recall_score(actual_vals, pred_vals))

if __name__ == '__main__':
	main()
