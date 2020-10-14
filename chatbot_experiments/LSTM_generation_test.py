#%%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import keras
from nltk.tokenize import word_tokenize, TweetTokenizer


import tensorflow.keras as keras
from tensorflow.keras.layers import Bidirectional, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM, Embedding
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam


import nltk
from nltk.corpus import stopwords
# print(stopwords.words('english'))

punctuation_list = ['.', ':', '!', '?', ',']
# import spacy
# nlp = spacy.load('en')

'''
1. tokenize the sentences, use SOS and EOS and UNK tokens. 
2. pass through an embedding layer
3. encode
4. pass through an embedding layer
5. decode
6. cross entropy loss


'''
#%%
df = pd.read_csv("../data/customer_service/twcs/twsc_sorted.csv")#, parse_dates=[3])
# df = pd.read_csv("data/customer_service/twcs/twsc_sorted.csv", parse_dates=[3])

#%%

def arr_to_str(array):
    _ = ""
    for f in array:
        _ = _ + f"{f} "
    _ = _[:-1]

    return _

def purge_at(text):
    text = text.split(" ")[1:]
    text = arr_to_str(text)
    return text

def purge_stopwords(text):
    text = set([word.lower() for word in text])
    b = set(stopwords.words('english') + punctuation_list)

    return list(text.difference(b))

    # new_sent = []
    # for word in text:
    #     if word in stopwords.words('english') or word in punctuation_list:
    #         continue
    #     new_sent.append(word)
    # return new_sent
#%%

amazon = df[df.author_id == 'AmazonHelp']

responses = amazon[~ pd.isna(amazon.in_response_to_tweet_id) & ~( amazon.inbound)]
responses.text = responses.text.apply(purge_at)

#%%

tic = time.time()
responses = responses.iloc[:500]
in_resp_to_list = list(responses.in_response_to_tweet_id)

queries = df[df.tweet_id.isin(in_resp_to_list)]
query_id_list = list(queries.tweet_id)

responses = responses[responses.in_response_to_tweet_id.isin(query_id_list)]
responses = responses.drop_duplicates("in_response_to_tweet_id")

q = list(queries.text.apply(purge_at))
r = list(responses.text)

toc = time.time()
print(f"Job took {toc-tic} seconds")

#%%

# #Lemmatize the queries.

# def lemmatize_(sent):
#     sent = arr_to_str(sent)
#     doc = nlp(sent)
#     return [token.lemma_ for token in doc]
#%%

tokenizer = TweetTokenizer().tokenize

tic = time.time()
# Split up sentences
q = list(map(tokenizer, q))
r = list(map(tokenizer, r))

toc = time.time()
print(f"Job took {toc-tic} seconds")

#%%
#Purge stopwords
## TODO: can you improve the execution time for purging stopwords
tic = time.time()
q = list(map(purge_stopwords, q))

toc = time.time()
print(f"Job took {toc-tic} seconds")

#%%

def build_dictionary(q, r):
    corpus = q + r

    corpus = [list(map(str.lower, line)) for line in corpus]
    vocab_list = np.unique(np.concatenate((corpus)))
    
    vocab = dict(zip(np.unique(np.concatenate((corpus))), range(len(vocab_list))))

    vocab["<SOS>"] = len(vocab_list) + 0
    vocab["<EOS>"] = len(vocab_list) + 1
    vocab["<UNK>"] = len(vocab_list) + 2

    return vocab

#%%

vocab = build_dictionary(q, r)
print(len(vocab))

#%%

## Now, transform everything in the input sequence to start with <SOS> and end
# with <EOS>, then tokenize everything numerically.

def append_sent_tokens(sent):
    sent = ['<SOS>', *sent, '<EOS>']
    return sent

#%%
q = list(map(append_sent_tokens, q))
r = list(map(append_sent_tokens, r))

#%%
def tokenize_sent(word):
    # tokenize_sentence = lambda x: vocab.get(x.lower())
    token = vocab.get(word.lower()) if word not in ['<SOS>', '<EOS>'] else vocab.get(word)

    if token == None:
        token = vocab.get("<UNK>")

    return token

#%%
q = [list(map(tokenize_sent, sent)) for sent in q]
r = [list(map(tokenize_sent, sent)) for sent in r]

#%%

# Now begins the padding phase. Pad with <EOS> token until same length.

def find_max(corpus):
        
    max_ = 0
    # corpus = q+r

    for sent in corpus:
        if len(sent) > max_:
            max_ = len(sent)

    # If it's an odd number just make it even to help us in model training.
    if max_ % 2 != 0:
        max_ += 1
    return max_

#%%
max_ = find_max(q+r)

#%%

def pad_seq(sent):

    return np.pad(sent, (0, max_ - len(sent)), 'constant', constant_values=(0, vocab.get("<EOS>")))

def one_hot(sent):
    return keras.utils.to_categorical(sent, num_classes = len(vocab))
#%%
q = np.array(list(map(pad_seq, q)))
r = np.array(list(map(pad_seq, r)))

#%%

# x_train = q
# y_train = r
# Next we convert to one hot vectors.
tic = time.time()
x_train = np.array(list(map(one_hot, q)))
y_train = np.array(list(map(one_hot, r)))

toc = time.time()
print(f"Job took {toc-tic} seconds.")
print(f"x_train: {x_train.shape}\ny_train: {y_train.shape}")
#%%
# x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
# y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
#%%
# save our arrays

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
#%%
# tic = time.time()

# x_train = np.load("x_train.npy")
# y_train = np.load("y_train.npy")
# toc = time.time()
# print(f"Job took {toc-tic} seconds")
#%%

#%%

from tensorflow.keras.layers import Conv1D

batch_size = 64
vocab_size = len(vocab)
input_layer = Input( shape=(x_train.shape[1], x_train.shape[2]))
# emb = Embedding(vocab_size+1, 400, input_length = x_train.shape[1])(input_layer)
lstm = Bidirectional(LSTM(800, batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),  return_sequences=True,  dropout=0.2))(input_layer, training = True)
lstm = Bidirectional(LSTM(400,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True,dropout=0.2))(lstm, training = True)
lstm = Bidirectional(LSTM(200,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True,dropout=0.2))(lstm, training = True)
lstm = Bidirectional(LSTM(50,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True,dropout=0.2))(lstm, training = True)
lstm = Bidirectional(LSTM(200,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True,dropout=0.2))(lstm, training = True)
lstm = Bidirectional(LSTM(400,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True, dropout=0.2))(lstm, training = True)
lstm = Bidirectional(LSTM(800,  batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),return_sequences=True, dropout=0.2))(lstm, training = True)
# cnn = Conv1D(x_train.shape[2], 1, 1, activation='softmax')(lstm)
dense = Dense(x_train.shape[2], activation='softmax')(lstm)
# lstm = LSTM(x_train.shape[1], return_sequences=True, activation='relu')(lstm)

model = Model(input_layer, dense)


# vocab_size = len(vocab)
# input_layer = Input( shape=( 38, 1222))
# # emb = Embedding(vocab_size+1, 200)(input_layer)
# lstm = Bidirectional(LSTM(200,  return_sequences=True, dropout=0.2))(input_layer, training = True)
# lstm = Bidirectional(LSTM(200,  return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Bidirectional(LSTM(200,  return_sequences=True,dropout=0.2))(lstm, training = True)
# lstm = Dense(1222, activation='softmax')(lstm)

# model = Model(input_layer, lstm)
print(model.summary())

#%%

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse', 'accuracy'])

# %%
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('lstm_model4.hdf5', monitor='accuracy', verbose=1, save_best_only=True, mode='max')
batch_size=64
epochs=1000

model.fit(x=x_train, y=y_train,
        batch_size=batch_size, epochs=epochs,
        verbose=1, callbacks=[checkpoint])


#%%
model = keras.models.load_model("lstm_model2.hdf5")

#%%
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
#%%

def eval_sentence(seq):
    keys = list(vocab.keys())
    return keys[seq]

def remove_sent_tokens(sent):
    return sent.replace("<SOS>", "").replace("<EOS>", "")
    
def stringify(seq):
    maxes = np.argmax(seq, axis=1)
    converted = list(map(eval_sentence, maxes))
    string = arr_to_str(converted)
    string = remove_sent_tokens(string)

    return string

def tokenify(seq):
    seq = list(map(tokenizer, seq))
    seq = list(map(purge_stopwords, seq))
    seq = list(map(append_sent_tokens, seq))
    seq = [list(map(tokenize_sent, sent)) for sent in seq]
    seq = np.array(list(map(pad_seq, seq)))
    seq = np.array(list(map(one_hot, seq)))
    return seq
#%%

import numpy as np

preds = np.round(model.predict(x_train[:10]))
s_p = list(map(stringify, preds))
s_x = list(map(stringify, x_train))
s_y = list(map(stringify, y_train))


# %%

test_query = "my package has not arrived yet"

test_query = [test_query]
test_query = tokenify(test_query)
# #%%

test_preds = np.round(model.predict(test_query))
s_preds = list(map(stringify, test_preds))
s_preds


#%%

num_generate = 10

# Converting our start string to numbers (vectorizing)
test_query

# Empty string to store our results
text_generated = []

# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
temperature = 1.0
input_eval = tokenify(["I"])
import tensorflow as tf 

# Here batch size == 1
model.reset_states()
for i in range(num_generate):

    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    input_eval = eval_sentence(predicted_id)
    input_eval = tokenify([input_eval])


    text_generated.append(eval_sentence(predicted_id))

#(test_query + ''.join(text_generated))
text_generated

# %%
