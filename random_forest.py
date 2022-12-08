import numpy as np

def shuffle(list):
	np.random.shuffle(list)
	return np.array(list)

#importing files and data organizing

#lists for storing documents
from file_parsing import parse_file, rating_review_split

train_pos = parse_file('train_pos_reviews.txt')
train_neg = parse_file('train_neg_reviews.txt')
dev_pos = parse_file('dev_pos_reviews.txt')
dev_neg = parse_file('dev_neg_reviews.txt')

all_vocabulary = open("aclImdb_v1\\aclImdb\\imdb.vocab", encoding = 'utf-8').read().splitlines()

train_all = shuffle(train_pos + train_neg)
dev_all = shuffle(dev_pos + dev_neg)

train_ratings, train_reviews = rating_review_split(train_all)
dev_ratings, dev_reviews = rating_review_split(dev_all)
train_polarities = [rating >= 7 for rating in train_ratings]
dev_polarities = [rating >= 7 for rating in dev_ratings]

#tf-idf

from sklearn.feature_extraction.text import TfidfVectorizer
from grammatical_tools import stops, word_split

vectorizer = TfidfVectorizer(strip_accents = 'ascii', lowercase=True, tokenizer = word_split, preprocessor=None, stop_words=None, vocabulary = all_vocabulary)

X_train = vectorizer.fit_transform(train_reviews)
X_dev = vectorizer.transform(dev_reviews)
Y_train = train_polarities
Y_dev = dev_polarities
X = vectorizer.transform(train_reviews + dev_reviews)
Y = train_polarities + dev_polarities

#random forest

from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier with 100 trees
model = RandomForestClassifier()

# Train the model on training data
model.fit(X_train, Y_train)

# Score
print(model.score(X_dev, Y_dev))

# get feature (word) importances
print(model.feature_importances_)

#grid search can help us find the best parameters

from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid for the model
param_grid = {
	'n_estimators': [10, 100, 1000],
	'max_depth': [5, 10, 50, 100],
	'min_impurity_decrease': [0, 0.1, 1],
	'max_features': [1, 10, 100, 1000, None]
}

model_grid = RandomForestClassifier()

# Use GridSearchCV to search for the best hyperparameters
clf = GridSearchCV(model_grid, param_grid, cv=5)


clf.fit(X, Y)

# Print the best hyperparameters
print(f"Best hyperparameters: {clf.best_params_}. Score: {clf.best_score_:.2f}")

# Import the necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

# Create a gradient boosting classifier
clf = GradientBoostingClassifier()

# Train the classifier on the data
clf.fit(X_train, Y_train)

# Make predictions on new data
clf.score(X_dev, Y_dev)

import xgboost as xgb

# Create the XGBoost model
model = xgb.XGBClassifier()

# Train the model on the training data
model.fit(X_train, Y_train)

# Evaluate the model on the test data
accuracy = model.score(X_dev, Y_dev)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Contextual polarity

from sklearn.linear_model import LogisticRegression
import numpy as np
model = LogisticRegression()
model.fit(X_train, y_train)
feature_names = np.array(vectorizer.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print("Negative Words", feature_names[sorted_coef_index[:10]])
print("Positive Words", feature_names[sorted_coef_index[-10:]])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = [
	"Nearest Neighbors",
	"Linear SVM",
	"RBF SVM",
	"Gaussian Process",
	"Decision Tree",
	"Random Forest",
	"Neural Net",
	"AdaBoost",
	"Naive Bayes",
	"QDA",
]

classifiers = [
	KNeighborsClassifier(30),
	SVC(kernel="linear", C=0.025),
	SVC(gamma=2, C=1),
	GaussianProcessClassifier(1.0 * RBF(1.0)),
	DecisionTreeClassifier(max_depth=50),
	RandomForestClassifier(max_depth=50, n_estimators=100, max_features=10),
	MLPClassifier(alpha=1, max_iter=1000),
	AdaBoostClassifier(),
	GaussianNB(),
	QuadraticDiscriminantAnalysis(),
]


figure = plt.figure(figsize=(27, 9))
i = 1
# preprocess dataset, split into training and test part

X_normal = StandardScaler(with_mean=False).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.4, random_state=42
)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

scores = {}
# iterate over classifiers
for name, clf in zip(names, classifiers):
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print(name, score)
	scores[name] = score


# doc2vec embeddings (instead of TF-IDF)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Create a list of TaggedDocument objects
documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(train_docs)]

# Initialize the Doc2Vec model
model = Doc2Vec(vector_size=300, min_count=1, epochs=50)

# Build the vocabulary
model.build_vocab(documents)

# Train the model
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

# Generate vector representation for a document
X_train = [model.infer_vector(doc) for doc in train_docs]