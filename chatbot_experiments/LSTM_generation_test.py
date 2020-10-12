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


'''
1. tokenize the sentences, use SOS and EOS and UNK tokens. 
2. pass through an embedding layer
3. encode
4. pass through an embedding layer
5. decode
6. cross entropy loss



'''
#%%
df = pd.read_csv("../data/customer_service/twcs/twsc_sorted.csv", parse_dates=[3])
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

#%%

amazon = df[df.author_id == 'AmazonHelp']

responses = amazon[~ pd.isna(amazon.in_response_to_tweet_id) & ~( amazon.inbound)]
responses.text = responses.text.apply(purge_at)

#%%

tic = time.time()
responses = responses.iloc[:1000]
in_resp_to_list = list(responses.in_response_to_tweet_id)

queries = df[df.tweet_id.isin(in_resp_to_list)]
query_id_list = list(queries.tweet_id)

responses = responses[responses.in_response_to_tweet_id.isin(query_id_list)]

q = list(queries.text.apply(purge_at))
r = list(responses.text)

toc = time.time()
print(f"Job took {toc-tic} seconds")

#%%

tokenizer = TweetTokenizer().tokenize

tic = time.time()
# Split up sentences
q = list(map(tokenizer, q))
r = list(map(tokenizer, r))

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

max_ = 0
corpus = q+r

for sent in corpus:
    if len(sent) > max_:
        max_ = len(sent)

# If it's an odd number just make it even to help us in model training.
if max_ % 2 != 0:
    max_ += 1

#%%

def pad_seq(sent):

    return np.pad(sent, (0, max_ - len(sent)), 'constant', constant_values=(0, vocab.get("<EOS>")))

def one_hot(sent):
    return keras.utils.to_categorical(sent, num_classes = len(vocab))
#%%

q = np.array(list(map(pad_seq, q)))
r = np.array(list(map(pad_seq, r)))

#%%

# Next we convert to one hot vectors.
x_train = np.array(list(map(one_hot, q)))
y_train = np.array(list(map(one_hot, r)))

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


vocab_size = len(vocab)
input_layer = Input( shape=( 38, vocab_size))
# emb = Embedding(vocab_size+1, 200)(input_layer)
lstm = Bidirectional(LSTM(200,  return_sequences=True, dropout=0.2))(input_layer, training = True)
lstm = Bidirectional(LSTM(200,  return_sequences=True,dropout=0.2))(lstm, training = True)
lstm = Bidirectional(LSTM(200,  return_sequences=True,dropout=0.2))(lstm, training = True)
lstm = LSTM(vocab_size, activation='softmax')(lstm)

model = Model(input_layer, lstm)


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
checkpoint = ModelCheckpoint('lstm_model.hdf5', monitor='accuracy', verbose=1, save_best_only=True, mode='max')
batch_size=64
epochs=800

model.fit(x=x_train, y=y_train,
        batch_size=batch_size, epochs=epochs,
        verbose=1, callbacks=[checkpoint])


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
    seq = list(map(append_sent_tokens, seq))
    seq = [list(map(tokenize_sent, sent)) for sent in seq]
    seq = np.array(list(map(pad_seq, seq)))
    out_seq = np.array(list(map(one_hot, seq)))
    return out_seq
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

# %%
