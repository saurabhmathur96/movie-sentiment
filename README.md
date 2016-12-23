# Movie Sentiment

Determine how positive a Movie Review is by analyising the text.

This project combines 1 Dimensional Convolutions, 
Word Embeddings learned in an unsupervised manner and Long Short Term Memory networks
to predict if a movie review is positive.

I used the pretrained GloVe (Global Vectors for Word Representation) word embeddings by 
Stanford to greatly reduce training time.

## Data
[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

A set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 

## Model Structure


| Input        |
| ------------- |
| Word Embedding | 
| Convolution 1D |
| MaxPooling |
| Convolution 1D |
| MaxPooling | 
| Convolution 1D |
| LSTM |
| Global MaxPooling |
| Fully Connected | 

## Setup

### Data
Download the movie review dataset from [here](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
and extract to data directory.

### Pretrained Model
Download GloVe word embedding model from [here (822MB)](http://nlp.stanford.edu/data/glove.6B.zip)
Place the `glove6B.100d.txt` file in models directory.
