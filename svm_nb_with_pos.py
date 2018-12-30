
# Importing Libararies
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

text = dataset['text'].copy()

stop = set(stopwords.words('english'))

text = text.str.lower().str.split()

whitelist = ["n't", "not", "don't", "aren't", "couldn't", 
             "didn't", "doesn't", "hadn't", "hasn't", "haven't", 
             "isn't", "mightn't", "mustn't", "needn't", 
             "shouldn't", "wasn't", "weren't", "won't", "wouldn't"]

for idx, stop_word in enumerate(stop):
    if stop_word not in whitelist:
        text = text.apply(lambda x : [item for item in x if item not in stop_word])


# POS tagging
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
# word embedding
# =============================================================================


from sklearn.feature_extraction.text import TfidfVectorizer # for vectorizing words 
from sklearn.model_selection import train_test_split
# vectorizing text
vectorizer = TfidfVectorizer('english')
text_vector = vectorizer.fit_transform(text_feature2)


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