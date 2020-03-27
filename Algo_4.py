

# -*- coding: utf-8 -*-
"""
Copyright 2018 Infosys Ltd.
Use of this source code is governed by MIT license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.

@author: zineb.mezzour, mohan.ponduri
"""

#%%%
'''
IMPORTS
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt  
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.linear_model import SGDClassifier

#%%

'''
Get the excel, 
take out the spaces, 
show a graph with the scoring, 
create a new variable Z with the columns we want as features
'''

full_df = pd.read_excel ('Scoring_v3.xlsx')


print(full_df['Title/Tweet'].apply(lambda x: len(x.split(' '))).sum())

# How many relevant news in total

full_df['Useful'].value_counts().plot(kind='bar')

print(full_df['Useful'].value_counts())

#full_df['Relevance'].value_counts().plot(kind='bar')

z = full_df['Title/Tweet'] + full_df['Source/User']+ full_df['Keywords'] +full_df['Subjects/Views']

#%%
'''
Text Normalization : cleaning the text of any szmbols,stopwords,ect.
Apply it to z
'''


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

#full_df['Title/Tweet'] = full_df['Title/Tweet'].apply(clean_text)
z = z.apply(clean_text)

print(full_df['Title/Tweet'].apply(lambda x: len(x.split(' '))).sum())

#%%
'''
Split the data into train and test
'''


X_train, X_test, y_train, y_test = train_test_split(z, full_df['Relevance_fig'], test_size=0.3, random_state=42)


#%% 
'''
Use countvectorizer to create first a matric of vocabulary =3481).
Gives an identifier per word.
Count then how many time the word appear in the sentence/news.
All news will be transformed into a big matrice. Cf word freq.
Top words gives the words that appear the most. 

'''
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())

top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)


#%% TFID
'''
TFID gives the frequence of a word in the sentence. 
It takes into consideration the length of the news.
'''

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_cv)
X_train_tfidf.shape

X_test_tfidf = tfidf_transformer.transform(X_test_cv)


#%% 
'''
Mutlinomial Naive Bayes without TFID

'''

naive_bayes = MultinomialNB()

naive_bayes.fit(X_train_cv, y_train)


predictions_nb = naive_bayes.predict(X_test_cv)


print('Accuracy score: ', accuracy_score(y_test, predictions_nb))

print('Precision score: ', precision_score(y_test, predictions_nb,average='micro'))

print('Recall score: ', recall_score(y_test, predictions_nb, average='micro'))


cm = confusion_matrix(y_test, predictions_nb)

sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False, xticklabels=['Irrelevant', 'Relevant'], yticklabels=['Irrelevant', 'Relevant'])
plt.xlabel('true label')
plt.ylabel('predicted label')

my_tags=['Irrelevant','Relevant']

print(classification_report(y_test, predictions_nb,target_names=my_tags))

#%%

'''
Mutlinomial Naive Bayes with TFID

'''
naive_bayes = MultinomialNB()

naive_bayes.fit(X_train_tfidf, y_train)


predictions = naive_bayes.predict(X_test_tfidf)


print('Accuracy score: ', accuracy_score(y_test, predictions))

print('Precision score: ', precision_score(y_test, predictions,average='micro'))

print('Recall score: ', recall_score(y_test, predictions, average='micro'))



cm = confusion_matrix(y_test, predictions)

sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False, xticklabels=['Irrelevant', 'Relevant'], yticklabels=['Irrelevant', 'Relevant'])
plt.xlabel('true label')
plt.ylabel('predicted label')

print(classification_report(y_test, predictions,target_names=my_tags))
#%% 
'''
SVM
'''

clf_svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)

clf_svm.fit(X_train_tfidf, y_train)

predicted_svm = clf_svm.predict(X_test_tfidf)


print('Accuracy score: ', accuracy_score(y_test, predicted_svm))

print('Precision score: ', precision_score(y_test, predicted_svm,average='micro'))

print('Recall score: ', recall_score(y_test, predicted_svm, average='micro'))


cm = confusion_matrix(y_test, predicted_svm)

sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False, xticklabels=['Irrelevant', 'Relevant'], yticklabels=['Irrelevant', 'Relevant'])
plt.xlabel('true label')
plt.ylabel('predicted label')


print(classification_report(y_test, predicted_svm,target_names=my_tags))

#%%

'''
Logistic Regression 
'''
from sklearn.linear_model import LogisticRegression

logreg =  LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train_tfidf, y_train)


y_pred = logreg.predict(X_test_tfidf)


print('Accuracy score: ', accuracy_score(y_test, y_pred))

print('Precision score: ', precision_score(y_test, y_pred,average='micro'))

print('Recall score: ', recall_score(y_test, y_pred, average='micro'))


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False, xticklabels=['Irrelevant', 'Relevant'], yticklabels=['Irrelevant', 'Relevant'])
plt.xlabel('true label')
plt.ylabel('predicted label')


print(classification_report(y_test, y_pred,target_names=my_tags))



#%%

'''
Here we will predict the news !
First you need to extract the news from the excel file
Then zou will have to get the columns of interest
After that, you will apply clean text function
As well as countvectorizer and tfidf

Finalz predict and concat 
and SEND !
'''
'''
new_df = pd.read_excel ('MV.xlsx')
NEWS = new_df['Title/Tweet'] + new_df['Source/User']+ new_df['Keywords']
NEWS = NEWS.apply(clean_text)


NEWS_cv = cv.transform(NEWS)
NEWS_tfdi = tfidf_transformer.transform(NEWS_cv)
'''
'''
Here you wil have to decide if you predict with or without TFIDF. 
You choose that by looking at the accuracy.
Here as well you choose which algorithm you want to use to predict
'''
'''
predictions_news = naive_bayes.predict(NEWS_cv)
'''
