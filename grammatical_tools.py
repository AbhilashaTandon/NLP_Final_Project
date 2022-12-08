from re import sub #used for fixing line breaks
import numpy as np
from nltk.corpus import stopwords
import nltk
stops = set(stopwords.words('english'))

stops_from_tokenization = set(["'d", "'ll", "'re", "'s", "'ve", 'could', 'might', 'must', 'need', 'sha', 'wo', 'would'])
negative_words = set(["n't", "nobody", "nothing", "never", "no", "won't", "not", "don't", "isn't", "can't", "wouldn't", "didn't", "ain't", "shouldn't", "couldn't", "mustn't"])
stops = (stops | stops_from_tokenization) - negative_words

#print(stops)
#list of common words that we dont need to calculate average scores for 

from nltk.tag import pos_tag

def tokenizer(text):
	tokenized = [word for word in nltk.word_tokenize(text)]
	return (tokenized, [("" if word in stops else word) for word in tokenized])

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

def negation_analyzer(text): #predicts what words are negated semantically
	splitted = word_split(text)
	polarities = [1 for word in splitted]
	for idx, word in enumerate(splitted):
		if(word in stops or word in negative_words):
			polarities[idx] = 0
		if(idx != 0):
			if(splitted[idx-1] in negative_words):
				polarities[idx] *= -1
	word_pols = {word: 0 for word in set(splitted)}
	for pol, word in zip(polarities, splitted):
		word_pols[word] += pol
	return word_pols

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
		splitted = set(word_split(sent[1]))
		if(splitted | negative_words != set()):
			print(sent)

if __name__ == '__main__':
	main()

