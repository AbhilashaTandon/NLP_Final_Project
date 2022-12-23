with open(f'train_neg_reviews.txt',encoding='utf-8') as f:
	contents = f.read()
	train_neg_reviews = [review[len('4\t'):] for review in contents.split('\n')]

with open(f'train_pos_reviews.txt',encoding='utf-8') as f:
	contents = f.read()
	train_pos_reviews = [review[len('4\t'):] for review in contents.split('\n')]

with open(f'test_neg_reviews.txt',encoding='utf-8') as f:
	contents = f.read()
	test_neg_reviews = [review[len('4\t'):] for review in contents.split('\n')]

with open(f'test_pos_reviews.txt',encoding='utf-8') as f:
	contents = f.read()
	test_pos_reviews = [review[len('4\t'):] for review in contents.split('\n')]
	
train_docs = train_neg_reviews + train_pos_reviews
y_train = [0]*len(train_neg_reviews) + [1]*len(train_pos_reviews)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(train_docs)
X_train = vectorizer.transform(train_docs)

test_docs = test_neg_reviews + test_pos_reviews
y_test = [0]*len(test_neg_reviews) + [1]*len(test_pos_reviews)
X_test = vectorizer.transform(test_docs)

X = vectorizer.transform(train_docs+test_docs)
y = y_train + y_test

print(X_train.shape, len(y_train), X_test.shape, len(y_test), X.shape, len(y))

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
from sklearn.metrics import classification_report

names = [
	#"Nearest Neighbors",
	#"Linear SVM",
	#"RBF SVM",
	"Gaussian Process",
	"Decision Tree",
	"Random Forest",
	"Neural Net",
	"AdaBoost",
	"Naive Bayes",
	"QDA",
]

classifiers = [
	#KNeighborsClassifier(30),
	#SVC(kernel="linear", C=0.025),
	#SVC(gamma=2, C=1),
	GaussianProcessClassifier(1.0 * RBF(1.0)),
	DecisionTreeClassifier(max_depth=50),
	RandomForestClassifier(max_depth=50, n_estimators=100, max_features=10),
	MLPClassifier(alpha=1, max_iter=1000),
	AdaBoostClassifier(),
	GaussianNB(),
	QuadraticDiscriminantAnalysis(),
]

#figure = plt.figure(figsize=(27, 9))
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
	clf.fit(X_train.toarray(), y_train)
	#score = clf.score(X_test, y_test)
	print(name)#, score)
	#scores[name] = score
	y_pred = clf.predict(X_test)
	print(classification_report(y_test, y_pred))




