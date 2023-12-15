# coding: utf-8

# In[1]:

import pandas as pd

# Loading the datasets
train = pd.read_csv('Dataset/fraudTrain.csv')
test = pd.read_csv('Dataset/fraudTest.csv')
train.head()


# In[2]:

def preprocess(DF):
    #Removing redundant columns
    columns = ['Unnamed: 0', 'cc_num', 'merchant', 'first', 'last', 'street', 'city','state', 'zip', 'job', 'trans_num', 'unix_time']
    DF.drop(columns, axis='columns', inplace=True)

    # Calculating age from transaction date and DOB
    DF['trans_date_trans_time']=pd.to_datetime(DF['trans_date_trans_time'])
    DF['trans_date']=DF['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
    DF['trans_date']=pd.to_datetime(DF['trans_date'])
    DF['dob']=pd.to_datetime(DF['dob'])
    DF['age'] = (DF['trans_date']-DF['dob']).dt.days / 365
    DF['trans_month'] = DF['trans_date'].dt.month
    DF['trans_year'] = DF['trans_date'].dt.year

    # Calculating distance between credit card holder and merchant
    DF['lat_distance'] = abs(DF['merch_lat']-DF['lat'])
    DF['long_distance'] = abs(DF['merch_long']-DF['long'])

    # Encoding gender values
    DF['gender'] = DF['gender'].map(lambda x : 1 if x == 'M' else 0)
    DF['gender']=DF['gender'].astype(int)

    # Encoding category values
    DF = pd.get_dummies(DF, columns=['category'])

    columns = ['trans_date_trans_time', 'lat', 'long', 'dob', 'merch_lat', 'merch_long', 'trans_date']
    DF.drop(columns, axis='columns', inplace=True)
    
    return DF


# In[3]:

# Preprocessing the train dataset
train_pre = preprocess(train.copy())
train_pre.head()


# In[4]:

# Preprocessing the test dataset
test_pre = preprocess(test.copy())
test_pre.head()


# In[5]:

from sklearn.model_selection import train_test_split

# Splitting the train dataset into training and testing dataset in the ratio 3:1
X_train, X_test, Y_train, Y_test = train_test_split(train_pre.drop(['is_fraud'], axis='columns'), train_pre['is_fraud'], test_size = 0.25, random_state = 45)


# In[6]:

from sklearn.linear_model import LogisticRegression

# Using Logistic Regression classifier
LR_model = LogisticRegression()
LR_model.fit(X_train, Y_train)


# In[7]:

from sklearn.metrics import accuracy_score, classification_report

Y_predict = LR_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_predict)
print(accuracy)
report = classification_report(Y_test, Y_predict, zero_division=1)
print(report)


# In[8]:

# Testing Logistic Regression model on test data set
test_predict = LR_model.predict(test_pre.drop(['is_fraud'], axis='columns'))

# Observing performance of model on the given dataset
accuracy = accuracy_score(test_pre['is_fraud'], test_predict)
print("Accuracy of model: ", accuracy)
report = classification_report(test_pre['is_fraud'], test_predict, zero_division=1)
print("Classification Report:\n", report)


# In[9]:

from sklearn.tree import DecisionTreeClassifier

# Using Decision Tree classifier
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, Y_train)


# In[10]:

Y_predict = DT_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_predict)
print(accuracy)
report = classification_report(Y_test, Y_predict, zero_division=1)
print(report)


# In[11]:

# Testing Decision Tree model on test data set
test_predict = DT_model.predict(test_pre.drop(['is_fraud'], axis='columns'))

# Observing performance of model on the given dataset
accuracy = accuracy_score(test_pre['is_fraud'], test_predict)
print("Accuracy of model: ", accuracy)
report = classification_report(test_pre['is_fraud'], test_predict, zero_division=1)
print("Classification Report:\n", report)


# In[12]:

from sklearn.ensemble import RandomForestClassifier

# Using Random Forest classifier
RF_model = RandomForestClassifier(n_estimators = 50)
RF_model.fit(X_train, Y_train)


# In[13]:

Y_predict = RF_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_predict)
print(accuracy)
report = classification_report(Y_test, Y_predict, zero_division=1)
print(report)


# In[14]:

# Testing Random Forest model on test data set
test_predict = RF_model.predict(test_pre.drop(['is_fraud'], axis='columns'))

# Observing performance of model on the given dataset
accuracy = accuracy_score(test_pre['is_fraud'], test_predict)
print("Accuracy of model: ", accuracy)
report = classification_report(test_pre['is_fraud'], test_predict, zero_division=1)
print("Classification Report:\n", report)
