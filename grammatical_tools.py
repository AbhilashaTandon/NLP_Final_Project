from re import sub #used for fixing line breaks
import numpy as np
from nltk.corpus import stopwords
import nltk
stops = set(stopwords.words('english'))

stops_from_tokenization = set(["'d", "'ll", "'re", "'s", "'ve", 'could', 'might', 'must', 'need', 'sha', 'wo', 'would'])
negative_words = set(["nobody", "nothing", "never", "no", "won't", "not", "don't", "isn't", "can't", "wouldn't", "didn't", "ain't", "shouldn't", "couldn't", "mustn't"])
stops = (stops | stops_from_tokenization) - negative_words

#print(stops)
#list of common words that we dont need to calculate average scores for 

from nltk.tag import pos_tag

def tokenizer(text):
	tokenized = [word for word in nltk.word_tokenize(text)]
	return (tokenized, [[word,''][word in stops] for word in tokenized])

def word_split(text):
	return [word for word in text.split() if word not in stops]

def chunker(sentence, regex):
	tokenized = nltk.word_tokenize(sentence)
	print(tokenized)
	
	pos_tagged = nltk.pos_tag(tokenized)
	print(pos_tagged)
	
	chunker = nltk.RegexpParser(regex)
	chunked = chunker.parse(pos_tagged)
	print(chunked)

from file_parsing import parse_file

def shuffle(list):
	np.random.shuffle(list)
	return list

#lists for storing documents
train_pos = parse_file('train_pos_reviews.txt')
train_neg = parse_file('train_neg_reviews.txt')

#shuffles together positive and negative reviews
train_all = shuffle(train_pos + train_neg)

def main():
	for sent in train_all[:100]:
		print(word_split(sent[1]))

if __name__ == '__main__':
	main()

