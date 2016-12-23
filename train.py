import os
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv1D, Dense, Embedding, Flatten, Input, LSTM, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#
# Configuration
#
MAX_NB_WORDS=25000
MAX_SEQUENCE_LENGTH=1000
N_GLOVE_TOKENS=400000
EMBEDDING_DIM = 100

#
# Load the data
#
positive_dir = "data/aclImdb/train/pos"
negative_dir = "data/aclImdb/train/neg"

def read_text(filename):
        with open(filename) as f:
                return f.read().lower()

print ("Reading negative reviews.")
negative_text = [read_text(os.path.join(negative_dir, filename))
        for filename in tqdm.tqdm(os.listdir(negative_dir))]
        
print ("Reading positive reviews.")
positive_text = [read_text(os.path.join(positive_dir, filename))
        for filename in tqdm.tqdm(os.listdir(positive_dir))]


labels_index = { "negative": 0, "positive": 1 }

labels = [0 for _ in range(len(negative_text))] + \
        [1 for _ in range(len(negative_text))]
 
texts = negative_text + positive_text
 


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np_utils.to_categorical(np.asarray(labels))
print ("data.shape = {0}, labels.shape = {1}".format(data.shape, labels.shape))

x_train, x_test, y_train, y_test = train_test_split(data, labels)


#
# Load word embeddings
#
print("Loading word embeddings.")
embeddings_index = dict()
with open("models/glove.6B.100d.txt") as f:
        for line in tqdm.tqdm(f, total=N_GLOVE_TOKENS):
                values = line.split()
                word, coefficients = values[0], np.asarray(values[1:], dtype=np.float32)
                embeddings_index[word] = coefficients

embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

print ("embedding_matrix.shape = {0}".format(embedding_matrix.shape))

embedding_layer = Embedding(len(word_index)+1,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)


#
# Build 1D ConvNet
#
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
embedded_sequences = embedding_layer(sequence_input)


x = Conv1D(128, 5, activation="relu")(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation="relu")(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation="relu")(x)


x = LSTM(64, dropout_W=0.2, dropout_U=0.2)(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation="relu")(x)

preds = Dense(len(labels_index), activation="softmax")(x)

model = Model(sequence_input, preds)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])

#
# Train the model
#
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          nb_epoch=4, batch_size=128)


model.save("models/convnet.h5")















