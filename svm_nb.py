
# Importing Libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer # for stemming of words
from nltk.corpus import stopwords # from stopwords removal 
from sklearn.feature_extraction.text import TfidfVectorizer # for vectorizing words 
from sklearn.model_selection import train_test_split # spliting dataset


# Importing Dataset
dataset = pd.read_csv('spam.csv', encoding = 'latin-1')

dataset.head()


'''As the preview of the data above shows there are three useless columns, 
these should be removed. I will also rename the remaining columns as "label" and "text" are not descriptive'''

dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset = dataset.rename(columns={"v1":"label", "v2":"text"})
dataset.head()


# get the Length of text in every instance and store it in 'length' column
dataset['length'] = dataset['text'].apply(len)
dataset.head()


# =============================================================================
# Preprocessing
# =============================================================================
''' 
Preprocessing must be performed on data to build an efficient classification model. 
First stopwords are removed (words that does not provide any meaning, hence even after removing them 
meaning of sentence doesn't change. e.g:- 'the','a','an','or' etc.). 
Then stemming on each word is performed (every word is replaced with the root of that word, 
for example "driving" or "drove" would become "drive".
'''

def preprocess(text1):
    
    text1 = text1.translate(str.maketrans('', '', string.punctuation))
    text1 = [word for word in text1.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text1:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words



# =============================================================================
# word embedding
# =============================================================================
''' TFIDF vectoriser is used to tranfrom the textual data into numerica fromat '''

# Copying textual data in 'text_feature'
text_feature = dataset['text'].copy()

# perfomr preprocessing on textual data
text_feature = text_feature.apply(preprocess)

# vectorizing text
vectorizer = TfidfVectorizer('english')
text_vector = vectorizer.fit_transform(text_feature)


X = text_vector
y = dataset['label']
# Train Test split of dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    
# =============================================================================
# Model Bullding (SVM)
# =============================================================================
 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

model = SVC(kernel = 'sigmoid', gamma = 1.0)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
accuracy_score(y_test, prediction)


# =============================================================================
# Naive Bayes
# =============================================================================
 
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=0.2)
mnb.fit(X_train, y_train)
prediction = mnb.predict(X_test)
accuracy_score(y_test,prediction)