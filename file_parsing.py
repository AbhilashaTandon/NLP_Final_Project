from re import sub #used for fixing line breaks

def parse_doc(doc): #parses "doc", line in file containing a review and its score
#and returns those separately along with simplifying the review somewhat
	splitted = doc.split('\t')
	rating, review = int(splitted[0]), '\t'.join(splitted[1:]) #score and review separated by tab 
	#but also tabs still occur in the reviews
	review = review.lower() #this may cause issues with say Bush vs bush 
	#but it probably fixes more problems than it causes
	review = review.replace('<br /><br />', '\n') #replace line break symbol w/ actual line break
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

def rating_review_split(docs):
	ratings = [int(doc[0]) for doc in docs]
	reviews = [doc[1] for doc in docs]
	return (ratings, reviews)