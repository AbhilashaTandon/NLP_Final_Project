import re #used for adding line breaks
#import numpy as np
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
#list of common words that we dont need to calculate average scores for 
from sklearn.feature_extraction.text import TfidfVectorizer

train_pos, train_neg, test_pos, test_neg = [],[],[],[] #lists for storing documents

#open files
with open('train_pos_reviews.txt', mode = 'r', encoding='utf-8') as train_pos_file:
	train_pos = train_pos_file.readlines()

with open('train_neg_reviews.txt', mode = 'r', encoding='utf-8') as train_neg_file:
	train_neg = train_neg_file.readlines()

with open('test_pos_reviews.txt', mode = 'r', encoding='utf-8') as test_pos_file:
	test_pos = test_pos_file.readlines()

with open('test_neg_reviews.txt', mode = 'r', encoding='utf-8') as test_neg_file:
	test_neg = test_neg_file.readlines()

train_all = train_pos + train_neg
test_all = test_pos + test_neg

#returns dictionary with keys for each word and values with # of occurences
def word_freq(text):
	word_freq = {}
	for word in text:
		if(word in word_freq):
			word_freq[word] += 1
		else:
			word_freq[word] = 1
	return word_freq

def tf(word, word_freq):
	total_words = sum(word_freq.values())
	if(word not in word_freq):
		return 0
	return (word_freq[word]/total_words)

def idf(occurences, num_texts):
	return np.log((1 + num_texts) / (occurs + 1))

#split document into words
#maybe use nltk.word_tokenize()?
#we should probably also add stemming to this
def split_text(doc):
	doc = doc.split()
	return doc

#finds review and score given document and makes a few substitutions to review text to make segmentation easier
def get_review(doc):
	current_review = review_list[index]
	score, doc = current_review.split('\t')[0], ''.join(current_review.split('\t')[1:]).lower() #extracts score and separates it from document, lowercase might interfere with names vs words (Jack vs jack) but it will work for now
	#score and document on same line, separated by tab
	
	doc = doc.replace('<br /><br />', '\n') #replace line break symbol with actual line break
	doc = re.sub(r'([^A-Za-z0-9])',' \g<1> ', doc) #makes non alphanumeric characters separate tokens
	return (score, doc)

review_list = train_all #list of all docs (reviews + scores)

num_reviews = len(review_list)

word_scores = {} 
#sum of review scores for every doc where word occurs

word_occurs = {} 
#number of docs where a given word occurs
#used for calculating idf

for index, doc in enumerate(review_list): #
	score, review = get_review(doc)
	splitted = split_text(review)
	freq = word_freq(splitted)
	for word in splitted: #
		if(word in ['"', "'"]): continue
		if(word in stops): continue
		if(word not in word_scores):
			#word_scores[word] = int(score) * tf(word, freq) * idfs[word]
			word_scores[word] = (int(score), 1)
		else:
			score_sum, count = word_scores[word]
			word_scores[word] = (score_sum + int(score), count+1)
			#maybe change this to just 1 if positive review and -1 if negative
	for uniq_word in freq.keys(): #this loop doesnt count duplicate words
		if(uniq_word not in word_occurs):
			word_occurs[uniq_word] = 1
		else:
			word_occurs[uniq_word] += 1
			
for key, value in word_scores.items():
	print(key, value)

with open('word_scores.txt', mode = 'w', encoding='utf-8') as output:
	for word in word_scores:
		score, count = word_scores[word]
		output.write(f'{word}\t{score/count}\n')

