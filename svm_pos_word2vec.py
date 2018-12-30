
# Importing Libararies

import numpy as np
import pandas as pd
import re as regex
from nltk.corpus import stopwords
from nltk import pos_tag_sents
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


# Importing dataset
dataset = pd.read_csv('spam.csv', encoding = 'latin-1')

'''As the preview of the data above shows there are three useless columns, 
these should be removed. I will also rename the remaining columns as "label" and "text" are not descriptive'''

dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset = dataset.rename(columns={"v1":"label", "v2":"text"})
dataset.head()

# =============================================================================
# Data Preprocessing
# =============================================================================

# copying Textual data to variable 'text'
text = dataset['text'].copy() 


## Stopword removal
stop = set(stopwords.words('english'))

text = text.str.lower().str.split()

whitelist = ["n't", "not", "don't", "aren't", "couldn't", 
             "didn't", "doesn't", "hadn't", "hasn't", "haven't", 
             "isn't", "mightn't", "mustn't", "needn't", 
             "shouldn't", "wasn't", "weren't", "won't", "wouldn't"]

for idx, stop_word in enumerate(stop):
    if stop_word not in whitelist:
        text = text.apply(lambda x : [item for item in x if item not in stop_word])


## Part-of-Speech(POS) tagging
texts = text.tolist()
tagged_texts = pos_tag_sents(texts)


# New column POS with tagged words
dataset['POS'] = tagged_texts


# Maping to string for removing parantheses
dataset['POS2'] = [', '.join(map(str, x)) for x in dataset['POS']]
# Removing parantheses, commas, and quatation marks
for remove in map(lambda r: regex.compile(regex.escape(r)), ["(",")","'",","]):dataset["POS2"].replace(remove, "", inplace=True)

# Copying Textual data
text_feature2 = dataset['POS2'].copy()



# =============================================================================
# Data split
# =============================================================================
from sklearn.preprocessing import LabelEncoder
X = text_feature2
le = LabelEncoder()
y = dataset['label']
y = le.fit_transform(y)
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# =============================================================================
# #Building word2vec model
# =============================================================================


import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST') 


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec 
import gensim



n_dim = 200
model_w2v = Word2Vec(size=n_dim, min_count=1)
model_w2v.build_vocab([x.words for x in tqdm(x_train)])
model_w2v.train([x.words for x in tqdm(x_train)], epochs=model_w2v.iter, total_examples=model_w2v.corpus_count)
total_examples=model_w2v.corpus_count

# =============================================================================
# Sequence of words to sequence of vectors
# =============================================================================

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from sklearn.feature_extraction.text import TfidfVectorizer

print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))


''' to represent Whole senetece using 200 dimentional vector, weighted average vector is calculated 
by combining word2vec and tfid vector'''

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


n_dim = 200
from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)

 
# =============================================================================
# Model Bullding (SVM)
# =============================================================================
 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

model = SVC(kernel = 'sigmoid', gamma = 1.0)
model.fit(train_vecs_w2v, y_train)

prediction = model.predict(test_vecs_w2v)
accuracy_score(y_test, prediction)

