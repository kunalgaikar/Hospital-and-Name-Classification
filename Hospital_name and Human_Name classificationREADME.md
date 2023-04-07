# Hospital-and-Name-Classification
# Assignment Name :- Hospital_name and Human_Name classification.
Here we classify a given name into Human Name and Hospital Name classes.
 So, we use Deep learning - •	 character-level models.
# Required Libraries and their uses-
# First we need to import libraries like-
1--> Pandas for mostly used for data analysis tasks in Python .Pandas library works well for numeric, alphabets, and heterogeneous types of data simultaneously.

2--> NumPy for mostly used for working with Numerical values as it makes it easy to apply mathematical functions.

3--> tensorflow for Being an Open-Source library for deep learning and machine learning, TensorFlow plays a role in text-based applications, image recognition, voice search, and many more.

4--> Keras sequential model is suitable for analysis and comparison of simple neural network-oriented models which comprises layers and their associated data using top to bottom flow. It makes use of a single set of input as to value and a single set of output as per flow.

5-->Here we use LSTM because --
   #LSTM (Long Short-Term Memory) network is a type of RNN (Recurrent Neural Network) that is widely used for learning sequential data prediction problems. As every other neural network LSTM also has some layers which help it to learn and recognize the pattern for better performance.

6-->Dense layer is used for implementing a dense layer that involves the neurons receiving the input from all the previous neurons that help implement the neural networks.

7-->Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words. Ideally, an embedding captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space. An embedding can be learned and reused across models.

8-->Adam uses estimations of first and second moments of gradient to adapt the learning rate for each weight of the neural network.

9-->Tokenizer is used for Transforms each text in texts to a sequence of integers.

10-->pad_sequences are used to pad the sequences with the same length.

##In Hospital-and-Name-Classification respiratory I am giving Hospital and Human Name Classification.ipynb file for coding, you can just download or copy it and paste in Google colab and run the program.

Also I give you below program for same.Before running the program you need to download csv file (human_names.csv and Hospital.csv) also.In this I am build Deep Learning model which can generate new hospital names and Deep Learning model which can classify a given name into Human Name and Hospital Name classes.


import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer # Tokenizer is used for Transforms each text in texts to a sequence of integers.

from tensorflow.keras.preprocessing.sequence import pad_sequences # pad_sequences are used to pad the sequences with the same length

from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense, Embedding, LSTM

from tensorflow.keras.optimizers import Adam

# Load hospital names dataset
df = pd.read_csv('Hospital.csv')

# Tokenize hospital names
tokenizer = Tokenizer(lower=True)
tokenizer.fit_on_texts(df.hosp_name)
tokenized_hospitals = tokenizer.texts_to_sequences(df.hosp_name)
df.head()

# Set max sequence length
max_length = max([len(x) for x in tokenized_hospitals])

max_length

# Pad tokenized hospital names
padded_hospitals = pad_sequences(tokenized_hospitals, maxlen=max_length, padding='post')

# Define model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 64, input_length=max_length))
model.add(LSTM(256))
model.add(Dense(64))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Generate new hospital names
start_word = "Hospital"
start_word_tokenized = tokenizer.texts_to_sequences([start_word])[0]
start_word_padded = pad_sequences([start_word_tokenized], maxlen=max_length)
generated_word = model.predict(start_word_padded, verbose=0)[0]
generated_word_index = np.argmax(generated_word)
generated_word_tokenized = tokenizer.index_word[generated_word_index]

# Print generated hospital names
generated_hospitals = []
for i in range(10):
    start_word_tokenized = tokenizer.texts_to_sequences([generated_word_tokenized])[0]
    start_word_padded = pad_sequences([start_word_tokenized], maxlen=max_length)
    generated_word = model.predict(start_word_padded, verbose=0)[0]
    generated_word_index = np.argmax(generated_word)
    generated_word_tokenized = tokenizer.index_word[generated_word_index]
    generated_hospitals.append(generated_word_tokenized)

print(generated_hospitals)

import numpy as np
import pandas as pd

from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential


# Read the dataset
data = pd.read_csv('Hospital.csv')
data.head()


# Tokenize the words for each hospital name
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data['hosp_name'])

# Generate sequences for each hospital name
sequences = tokenizer.texts_to_sequences(data['hosp_name'])


# Get the vocabulary size
vocabulary_size = len(tokenizer.word_index)+1

# Generate X and y
X = pad_sequences(sequences, maxlen=50)
y = np.array([1] * X.shape[0])


# Create model
model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=50, input_length=50))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, batch_size=128, epochs=20, verbose=1)

# Generate new hospital names
seed_text = "Hosp"
for _ in range(20):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=50, padding='pre')
    predicted = (model.predict(token_list) > 0.5).astype("int32")
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += output_word

print(seed_text)



# making Data frame
data_human = pd.read_csv('human_names.csv')
data_hospital = pd.read_csv('Hospital.csv')

# Classifying Human and Hospital Names Using Character-Level Deep Learning

# Read the dataset
data_human = pd.read_csv('human_names.csv')
data_hospital = pd.read_csv('Hospital.csv')

# Tokenize the words for each name
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data_human.Name)
tokenizer.fit_on_texts(data_hospital.hosp_name)

# Generate sequences for each name
human_sequences = tokenizer.texts_to_sequences(data_human['Name'])
hospital_sequences = tokenizer.texts_to_sequences(data_hospital['hosp_name'])


# Test the model
seed_text = "jhon hospital"
token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences([token_list], maxlen=50, padding='pre')
predicted=(model.predict(token_list) > 0.5).astype("int32")
if predicted == 0:
    print("This is a human name")
else:
    print("This is a hospital name")
 
