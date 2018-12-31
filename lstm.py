
# Importing Libaraaries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

# Importing dataset
dataset = pd.read_csv("spam.csv", encoding = "latin-1")
dataset.head()


'''As the preview of the data shows there are three useless columns, 
these should be removed. I will also rename the remaining columns as "label" and "text" are not descriptive'''

dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset = dataset.rename(columns={"v1":"label", "v2":"text"})
dataset.head()


# Visualize the distribution
sns.countplot(dataset['label'])
plt.xlabel("Label")
plt.title('Number of hams and spams')


# Encoding the labels into '0' and '1'
X = dataset['text']
y = dataset['label']
le = LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1, 1)

# SPliting dataset into trainig and tesing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

''' Process the data
Tokenize the data and convert the text to sequences.
Add padding to ensure that all the sequences have the same shape.'''

max_words = 1000
max_len = 150
tok = Tokenizer(num_words = max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen = max_len)

# =============================================================================
# RNN(LSTM)
# =============================================================================

model = Sequential()
model.add(keras.layers.InputLayer(input_shape = [max_len]))
model.add(keras.layers.Embedding(max_words, 50 , input_length = max_len)) #for Word embedding (text to vector)
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(units = 256, name = 'FC1', activation='relu'))
model.add(Dropout(0.5)) #                                                 To overcome the posibility of overfitting
model.add(keras.layers.Dense(units=1, name = 'Out_layer', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Fitting model to training data
model.fit(sequences_matrix, y_train, batch_size = 128, epochs=10,
          validation_split = 0.2, callbacks = [EarlyStopping(monitor = 'val_loss', min_delta = 0.0001)])


# test data
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix, y_test)


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
