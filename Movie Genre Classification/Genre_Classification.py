# coding: utf-8
# In[1]:

import re
import pandas as pd

train_data_path = 'Genre Classification Dataset/train_data.txt'
test_data_path = 'Genre Classification Dataset/test_data.txt'
test_data_soln_path = 'Genre Classification Dataset/test_data_solution.txt'

train = pd.read_csv(train_data_path, sep=':::', engine='python', names=['index', 'movie_title', 'genres', 'plot_summary'])
test = pd.read_csv(test_data_path, sep=':::', engine='python', names=['index', 'movie_title', 'plot_summary'])
test_solution = pd.read_csv(test_data_soln_path, sep=':::', engine='python', names=['index', 'movie_title', 'genres', 'plot_summary'])

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

# Cleaning the plot summary text
train['clean_plot_summary'] = train['plot_summary'].apply(clean_text)
test['clean_plot_summary'] = test['plot_summary'].apply(clean_text)


# In[2]:

genre_mapping = {
    'drama': 1,
    'documentary': 2,
    'comedy': 3,
    'short': 4,
    'horror': 5,
    'thriller': 6,
    'action': 7,
    'western': 8,
    'reality-tv': 9,
    'family': 10,
    'adventure': 11,
    'music': 12,
    'romance': 13,
    'sci-fi': 14,
    'adult': 15,
    'crime': 16,
    'animation': 17,
    'sport': 18,
    'talk-show': 19,
    'fantasy': 20,
    'mystery': 21,
    'musical': 22,
    'biography': 23,
    'history': 24,
    'game-show': 25,
    'news': 26,
    'war': 27
}

# Encoding the movie genres and removing the irrelevant columns of train dataset
train['genres'] = train['genres'].str.strip()
train.dropna(subset=['genres'], inplace=True)
train['genres_encoded'] = train.genres.map(genre_mapping)
train.drop(['movie_title', 'index', 'plot_summary', 'genres'], axis=1, inplace=True)
train.head()


# In[3]:

# Removing the irrelevant columns of test dataset
test.drop(['movie_title', 'index', 'plot_summary'], axis=1, inplace=True)
test.head()


# In[4]:

# Encoding the movie genres and removing the irrelevant columns of test solution dataset
test_solution['genres'] = test_solution['genres'].str.strip()
test_solution.dropna(subset=['genres'], inplace=True)
test_solution['genres_encoded'] = test_solution.genres.map(genre_mapping)
test_solution.drop(['movie_title', 'index', 'plot_summary', 'genres'], axis=1, inplace=True)
test_solution.head()


# In[5]:

# Exploring the occuring frequencies of different movie genres
train.genres_encoded.value_counts()


# In[6]:

# Splitting the train dataset into training and testing dataset in the ratio 3:1
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train['clean_plot_summary'], train['genres_encoded'], test_size=0.25, random_state=47, stratify=train.genres_encoded) #25% of data to be used as the test set


# In[7]:

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization of Training set and Testing set
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[8]:

from sklearn.naive_bayes import MultinomialNB

# Using Naive Bayes classifier
naiveBayes_model = MultinomialNB()
naiveBayes_model.fit(X_train_tfidf, Y_train)


# In[9]:

from sklearn.metrics import accuracy_score, classification_report

y_predicted = naiveBayes_model.predict(X_test_tfidf)
accuracy = accuracy_score(Y_test, y_predicted)
print(f'Accuracy: {accuracy:.2f}')

classification_rep = classification_report(Y_test, y_predicted, zero_division=1)
print('Classification Report:\n', classification_rep)


# In[10]:

# Testing Naive Bayes model on test data set
test_tfidf = tfidf_vectorizer.transform(test['clean_plot_summary'])
predicted_result = naiveBayes_model.predict(test_tfidf)
test_solution['genres_encoded'] = test_solution['genres_encoded'].astype('int64')

# Observing performance of model on the given dataset
accuracy = accuracy_score(test_solution['genres_encoded'], predicted_result)
print(f'Accuracy of Tested solution: {accuracy:.2f}')
classification_rep = classification_report(test_solution['genres_encoded'], predicted_result, zero_division=1)
print('Classification Report:\n', classification_rep)


# In[11]:

from sklearn.linear_model import LogisticRegression

# Using Logistic Regression classifier
logistic_regression_model = LogisticRegression(max_iter=400)
logistic_regression_model.fit(X_train_tfidf, Y_train)


# In[12]:

from sklearn.metrics import accuracy_score, classification_report
y_predicted = logistic_regression_model.predict(X_test_tfidf)
accuracy = accuracy_score(Y_test, y_predicted)
print(f'Accuracy: {accuracy:.2f}')

classification_rep = classification_report(Y_test, y_predicted, zero_division=1)
print('Classification Report:\n', classification_rep)


# In[13]:

# Testing Logistic Regression model on test data set
test_tfidf = tfidf_vectorizer.transform(test['clean_plot_summary'])
predicted_result = logistic_regression_model.predict(test_tfidf)

# Observing performance of model on the given dataset
accuracy = accuracy_score(test_solution['genres_encoded'], predicted_result)
print(f'Accuracy of Tested solution: {accuracy:.2f}')
classification_rep = classification_report(test_solution['genres_encoded'], predicted_result, zero_division=1)
print('Classification Report:\n', classification_rep)


# In[14]:

from sklearn.svm import SVC

# Using support vector classifier
SVM_model = SVC(random_state=1)
SVM_model.fit(X_train_tfidf, Y_train)


# In[15]:

from sklearn.metrics import accuracy_score, classification_report
y_predicted = SVM_model.predict(X_test_tfidf)
accuracy = accuracy_score(Y_test, y_predicted)
print(f'Accuracy: {accuracy:.2f}')

classification_rep = classification_report(Y_test, y_predicted, zero_division=1)
print('Classification Report:\n', classification_rep)


# In[16]:

# Testing SVM model on test data set
test_tfidf = tfidf_vectorizer.transform(test['clean_plot_summary'])
predicted_result = SVM_model.predict(test_tfidf)

# Observing performance of model on the given dataset
accuracy = accuracy_score(test_solution['genres_encoded'], predicted_result)
print(f'Accuracy of Tested solution: {accuracy:.2f}')
classification_rep = classification_report(test_solution['genres_encoded'], predicted_result, zero_division=1)
print('Classification Report:\n', classification_rep)
