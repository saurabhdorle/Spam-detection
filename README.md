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

4) Part-of-Speech(POS) tagging: Tags are assigned to every word, whether that word is noun, pronoun, verb etc. So that it will help in identifying the maeanig behind while sentence.
e.g.: - 'John' 'NN' (NN = Noun i.e. tag)

5) word embedding: Textual data transformed into numeric format using TFIDF(Term Frequency - Inverse Document frequency)

6) Word Embedding using word2vec: Word2vec is word embedding model developed by google. It is shallow neural network base on skip gram model. Here, Word2vec model is trained using 'spam' dataset for representing every word with 200 dimensional vector. 
e.g.:- to represent the sentence of size 6 words, total 1200 dimension vector is required, which is pretty huge.
To reduce the vector size as well as maintain the context between words, weighted average vector of word2vec and TFIDF matrix is calculated. Now every sentence is represented using 200 dimentional vector only.

## Classification Models:
1) SVM(Support Vector Machine) : 98.20 accruracy
2) Naive Bayes : 97.82 accruracy
3) LSTM(Long Short Term Memory) : 98.70 accruracy
4) SVM with word2vec and tfidf combined : 98.65 accruracy
