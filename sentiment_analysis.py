#!/usr/bin/env python
# coding: utf-8

# ### Data Wrangling

# In[1]:


#from google.colab import drive

#drive.mount('/content/gdrive', force_remount=True)

#root_dir = "/content/gdrive/My Drive/NLP Sentiment Analysis"


# In[2]:


#with open(f'{root_dir}/train_neg_reviews.txt') as f:
with open(f'train_neg_reviews.txt') as f:
  contents = f.read()
  train_neg_reviews = [review for review in contents.split('\n')]

#with open(f'{root_dir}/train_pos_reviews.txt') as f:
with open(f'train_pos_reviews.txt') as f:
  contents = f.read()
  train_pos_reviews = [review for review in contents.split('\n')]

#with open(f'{root_dir}/test_neg_reviews.txt') as f:
with open(f'test_neg_reviews.txt') as f:
  contents = f.read()
  test_neg_reviews = [review for review in contents.split('\n')]

#with open(f'{root_dir}/test_pos_reviews.txt') as f:
with open(f'test_pos_reviews.txt') as f:
  contents = f.read()
  test_pos_reviews = [review for review in contents.split('\n')]


# In[3]:


import pandas as pd
reviews = train_neg_reviews + test_neg_reviews + train_pos_reviews + test_pos_reviews
scores = [int(review.split('\t')[0] or 3) for review in reviews]
reviews_text = [''.join(review.split('\t')[1:]) for review in reviews]
classification = [0]*len(train_neg_reviews + test_neg_reviews) + [1]*len(train_pos_reviews + test_pos_reviews)
df = pd.DataFrame({'review': reviews_text, 'score': scores, 'classification': classification})
df = df.sample(frac=1, random_state=0) # shuffle


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(df.review)


# In[ ]:


tfidf_embeddings = vectorizer.transform(df.review)


# A random forest is a type of ensemble machine learning model that is made up of multiple decision trees. Ensemble models combine the predictions of multiple individual models to make more accurate predictions. In a random forest, each decision tree is trained on a random subset of the data, and the final prediction is made by averaging the predictions of all the individual decision trees.
# 
# Here is an example of how to train a random forest using the scikit-learn library in Python:
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_embeddings, df.classification, random_state=0)


# ## Model Fitting

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier with 100 trees
model = RandomForestClassifier()

# Train the model on training data
model.fit(X_train, y_train)

# Score
model.score(X_test, y_test)


# In[ ]:


from sklearn.metrics import f1_score, classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# ## Hyperparameter Searching

# In[ ]:


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


# clf.fit(X, y)

# Print the best hyperparameters
# print(f"Best hyperparameters: {clf.best_params_}. Score: {clf.best_score_:.2f}")


# ## Contextual Polarity

# In[ ]:


from sklearn.linear_model import LogisticRegression
import numpy as np
model = LogisticRegression()
model.fit(X_train, y_train)
feature_names = np.array(vectorizer.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print("Negative Words", feature_names[sorted_coef_index[:10]])
print("Positive Words", feature_names[sorted_coef_index[-10:]])


# ## More Models!

# In[ ]:


# Import the necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

# Create a gradient boosting classifier
clf = GradientBoostingClassifier()

# Train the classifier on the data
clf.fit(X_train, y_train)

# Make predictions on new data
clf.score(X_test, y_test)


# In[ ]:


import xgboost as xgb

# Create the XGBoost model
model = xgb.XGBClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# ## More Advanced Embeddings

# In[ ]:


# !pip install openai
# import openai
# openai.api_key = # GET THIS FROM JOSIAH IF NEEDED
# from openai.embeddings_utils import cosine_similarity, get_embeddings as _get_embeddings, get_embedding as _get_embedding
# get_embeddings = lambda x: _get_embeddings(x, 'text-embedding-ada-002')
# get_embedding = lambda x: _get_embedding(x, 'text-embedding-ada-002')
# sub = df.iloc[:2000]
# sub['ada_embeddings'] = get_embeddings(sub.review)
# sub.to_csv('embedded_reviews.csv', index=False)


# In[4]:


sub = pd.read_csv(root_dir + '/embedded_reviews.csv')
sub.ada_embeddings = sub.ada_embeddings.apply(eval)


# In[ ]:


X = np.array([emb for emb in sub.ada_embeddings.values])
X.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, sub.classification)
model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[ ]:


from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


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


clf.fit(X, sub.classification)

# Print the best hyperparameters
print(f"Best hyperparameters: {clf.best_params_}. Score: {clf.best_score_:.2f}")


# In[ ]:




