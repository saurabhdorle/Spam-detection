# Spam-detection

SMS Spam detection using dataset of 5.5k messages in English language.

## Dataset:
Data consists of toatl 5.5k messages in textual format, labels as ham(not-spam) and spam(as spam).
First column has labels, second column has text data, where remaining 3 columns does'nt contain any data.

## Prerpocessing:

Befor building classification model for the dataset, textual data needs to be preocessed to get the efficient results.
For this, various preprocessing tasks are performed.
1) Tokenization :  Sentence in split into seperate words

2) Stopword removal: Stopwords are the words which does not carry any meaning, e.g:- 'a', 'and, 'or', 'the' etc. 
So these words are removed using NLTK.

3) Lemmetization: Every single words is replaced by its root word.
e.g.:- 'tasted', 'tasting' replace by 'taste' only.

4) word embedding: Textual data transformed into numeric format using TFIDF(Term Frequency - Inverse Document frequency)

## Classification Models:
1) SVM(Support Vector Machine)
2) Naive Bayes
3) LSTM(Long Short Term Memory)
