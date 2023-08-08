#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


raw_mail_data = pd.read_csv('spam mail detection dataset.csv')


# In[3]:


print(raw_mail_data)


# In[4]:


mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[5]:


mail_data.head()


# In[6]:


mail_data.shape


# In[7]:


mail_data.loc[mail_data['v1'] == 'spam', 'v1',] = 0
mail_data.loc[mail_data['v1'] == 'ham', 'v1',] = 1


# In[8]:


X = mail_data['v2']

Y = mail_data['v1']


# In[9]:


print(X)


# In[10]:


print(Y)


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[12]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[13]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[14]:


print(X_train)


# In[15]:


print(X_train_features)


# In[16]:


model = LogisticRegression()


# In[17]:


model.fit(X_train_features, Y_train)


# In[18]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[19]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[20]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[21]:


print('Accuracy on test data : ', accuracy_on_test_data)


# In[22]:


input_mail = ["Even my brother is not like to speak with me. They treat me like aids patent."]

input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)
print(prediction)

if (prediction[0]==1):
   print('Ham mail')
else:
    print('Spam mail')

