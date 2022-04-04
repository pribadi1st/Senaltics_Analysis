from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import nltk

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import string
import random

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

#List of Function
def remove_mentions_hastaghs(text):
    pattern = r'[0-9]'
    text = text.lower()
    text = re.sub(pattern, '', text)
    text = text.replace('â™¥', 'heartemoticon')
    final = ' '.join(word for word in text.split(' ') if not word.startswith('@') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('#') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('www') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('http') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('ð') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('ñ') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('î') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('å') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('ã') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('â') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('è') )
    final = ' '.join(word for word in final.split(' ') if not word.startswith('zã') )
    final = ' '.join(word for word in final.split(' ') if not word.isdigit())
    final = ' '.join(word for word in final.split(' ') if word.isalnum() )
    return final


def numerized(text):
    if (text == 'Negative'):
        return 0
    return 1

def cleaning_stopwords(text):
    final = [word for word in text if word not in stop_words]
    return final

def remove_tokenize(text):
    final = " ".join(word_tokenize(text))
    translator = str.maketrans('', '', punctuations_list)
    remove_digits = str.maketrans('', '', digits_list)
    final = final.translate(remove_digits)
    final = final.translate(translator)
    return final

def remove_punct(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub('[0–9]+', '', text)
    return text

def tokenized_words(text):
    token = word_tokenize(text)
    return token

def stem_word(text):
    final = [st.stem(word) for word in text]
    return final

def lemmatize_data(text):
    final = [lm.lemmatize(word) for word in text]
    return " ".join(final)

def word_feature(words):
    my_dict= dict([(word, True) for word in words])
    return my_dict

def save_train_model():
    f = open('my_new_classifier.pickle', 'wb')
    pickle.dump(classifier, f)

def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.show()

#End of function list
neg_reviews = []
for words in twitter_samples.strings("negative_tweets.json"):
    neg_reviews.append(['Negative', words])
pos_reviews = []
for words in twitter_samples.strings("positive_tweets.json"):
    pos_reviews.append(['Positive', words])
total =  neg_reviews + pos_reviews
df = pd.DataFrame(total, columns = ['categories', 'text'])
df['numerized'] = df['categories'].apply(numerized)

#Remove mention and hastags
df['new_text'] = df['text'].apply(remove_mentions_hastaghs)
#Remove punctuation
df['non_punct'] = df['new_text'].apply(remove_punct)
#Tokenized words
df['tokenized'] = df['non_punct'].apply(tokenized_words)
#Remove Stopwords
#Define Stopwords
stop_words = set(stopwords.words('english'))
new_stop_words = {"rt", "i'd", "isn't", 'gt', 'nt', 'amp'}
stop_words = set.union(stop_words,new_stop_words)
#---
df['non_stopwords'] = df['tokenized'].apply(cleaning_stopwords)
#Stemming
st = nltk.PorterStemmer()
df['stem_words'] = df['non_stopwords'].apply(stem_word)
#Lemmatize data
lm = nltk.WordNetLemmatizer()
df['lem_words'] = df['stem_words'].apply(lemmatize_data)

X = df.lem_words

#Ganti ini
y = df.numerized
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state =26105111)
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

#print(y_train)
vect = CountVectorizer()
BoW = vect.fit_transform(df.text)

vocabulary = vect.vocabulary_

values = [] #in same order as traversing keys
keys = [] #also needed to preserve order
for key in vocabulary.keys():
  keys.append(key)
  values.append(vocabulary[key])

data = {'words': keys, 'total': values}
df2 = pd.DataFrame(data=data)
#df2.nlargest(n=10, columns=['total'])
#Model BNB
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)

#
fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()
#### END of BNB Model

#SVM Model
from sklearn.calibration import CalibratedClassifierCV

SVCmodel = LinearSVC()
SVCmodel = CalibratedClassifierCV(SVCmodel)
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

import pickle
# pickling the vectorizer
filename = 'N_Vectorizer.sav'
pickle.dump(vectoriser, open(filename, 'wb'))
# pickling the model SVM
filename = 'N_SVM_classifier.sav'
pickle.dump(SVCmodel, open(filename, 'wb'))
# Pickling the model NB
filename = 'N_NB_classifier.sav'
pickle.dump(BNBmodel, open(filename, 'wb'))
