#%%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import keras


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
# df = pd.read_csv("data/customer_service/twcs/twcs.csv", parse_dates=[3]).sort_values(by="created_at")
# df.to_csv("data/customer_service/twcs/twsc_sorted.csv", index=False)


#%%

amazon = df[df.author_id == 'AmazonHelp']

#%%

def loadGloveModel(File):
    tic = time.time()
    print("Loading Glove Model")
    f = open(File,'r', encoding="utf8")
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    toc = time.time()
    print(len(gloveModel)," words loaded!")
    print(f"Job took {toc-tic} seconds.")
    return gloveModel

#%%

glove_path = '../glove/glove.6B.300d.txt'
glove = loadGloveModel(glove_path)

#%%

def arr_to_str(array):
    _ = ""
    for f in array:
        _ = _ + f" {f} "
    _ = _[:-1]

    return _


#%%

# queries = df[pd.isna(df.in_response_to_tweet_id) & (df.inbound==True)]
# queries = df[(df.inbound==True)]
# responses = df[df.inbound == False]

responses = amazon[~ pd.isna(amazon.in_response_to_tweet_id) & ~( amazon.inbound)]

#%%

x = []
y = []

tic = time.time()
for f in range(responses.iloc[:1000].shape[0]):

    in_resp_to = responses.iloc[f].in_response_to_tweet_id
    query = df[df.tweet_id == in_resp_to]
    if len(query) == 0:
        continue

    response_str = str(responses.iloc[f].text).split(" ")[2:]
    response_str = arr_to_str(response_str)

    query_str = str(query.iloc[0].text).split(" ")[2:]
    query_str = arr_to_str(query_str)
    
    y.append(response_str)
    x.append(query_str)

toc = time.time()

print(f"Job took {toc-tic} seconds")
#%%

def process_token(token):
    replace_list = [".", "!", ",", "?", "'"]

    for r in replace_list:
        token = token.replace(r, "")
    
    return token.lower()
    
def tokenize_set(set):

    tic = time.time()

    set_ = []
    for line in range(len(set[:])):
        tokens = set[line].split(" ")

        line = []
        for token in tokens:
            cleaned_token = process_token(token)
            try:
                num = glove[cleaned_token]
            except:
                print(f"Unknown token {token}")
                continue
            line.append(num)

        set_.append(line)

    
    set_ = np.array(set_)
    toc = time.time()
    print(f"Job took {toc-tic} seconds")

    return set_

def add_padding(x_in, y_in):

    combined = np.concatenate((x_in, y_in))
    # Padding:
    max_ = 0
    for line in combined:
        length = len(line)
        if length > max_:
            max_ = length
    
    x_out = []
    y_out = []
    for f in range(len(x)):
        # x_in[f] = np.pad(x_in[f], (0, max_ - len(x_in[f])))
        x_out.append(np.pad(x_in[f], (0, max_ - len(x_in[f]))))
        # x_in[f] = x_in[f].reshape(1, x_in[f])

    for f in range(len(y)):
        # y_in[f] = np.pad(y_in[f], (0, max_ - len(y_in[f])))
        y_out.append(np.pad(y_in[f], (0, max_ - len(y_in[f]))))
        # y_in[f] = x_in[f].reshape(1, y_in[f])

    x_out = np.array(x_out)
    y_out = np.array(y_out)
    return x_out, y_out

# %%

# corpus = x + y 

# vocab_list = []

# tic = time.time()
# for line in corpus[:]:
#     tokens = line.split(" ")
#     for token in tokens:
#         if token not in vocab_list:
#             token = process_token(token)
#             vocab_list.append(token)
    
# vocab_list = np.unique(vocab_list)
# toc = time.time()
# print(f"Job took {toc-tic} seconds")

#%%
# vocab_dict = {}

# tic = time.time()

# for f in range(len(vocab_list)):
#     word = vocab_list[f]
#     vocab_dict[word] = f + 1

# toc = time.time()
# print(f"Job took {toc-tic} seconds")

#%%

x_set = tokenize_set(x)
y_set = tokenize_set(y)

#%%
x_set, y_set = add_padding(x_set, y_set)

#%%




# %%

# Convert to one hot vectors

oh_x_set = []

tic = time.time()

for f in range(len(x_set)):
    line = x_set[f]
    oh_x_line = []
    for g in range(len(line)):

        # one_hot = list(np.zeros((len(vocab_dict)+1)))
        # one_hot[line[g]] = 1
        # oh_x_line.append(one_hot)

        # oh_x_line.append(keras.utils.to_categorical(line[g], len(vocab_dict)+1))
    oh_x_set.append(oh_x_line)



toc = time.time()
print(f"Job took {toc-tic} seconds")

#%%

tic = time.time()

oh_x_set = np.array(oh_x_set)

toc = time.time()
print(f"Job took {toc-tic} seconds")
#%%

oh_y_set = []


tic = time.time()

for f in range(len(y_set)):
    line = y_set[f]
    oh_y_line = []
    for g in range(len(line)):

        # one_hot = list(np.zeros((len(vocab_dict)+1)))
        # one_hot[line[g]] = 1
        # oh_y_line.append(one_hot)

        oh_y_line.append(keras.utils.to_categorical(line[g], len(vocab_dict)+1))
    oh_y_set.append(oh_y_line)

toc = time.time()
print(f"Job took {toc-tic} seconds")

#%%
tic = time.time()
oh_y_set = np.array(oh_y_set)

toc = time.time()
print(f"Job took {toc-tic} seconds")
#%%
# x_set = x_set.reshape(x_set.shape[0], 1, x_set.shape[1])
# y_set = y_set.reshape(y_set.shape[0], 1, y_set.shape[1])
#%%

# Use GLOVE to convert every OH array into a feature vector.



#%% Saving these arrays to avoid the long processing times otherwise.

tic = time.time()

np.save("oh_x_set.npy", oh_x_set)
np.save("oh_y_set.npy", oh_y_set)

toc = time.time()
print(f"Job took {toc-tic} seconds")

#%%

oh_x_set = np.load("oh_x_set.npy")
oh_y_set = np.load("oh_y_set.npy")
#%%
# vectorizer = TfidfVectorizer()
# fit = vectorizer.fit(corpus)
# x = fit.transform(x).toarray()
# y = fit.transform(y).toarray()
# x = x.reshape(x.shape[0], 1, x.shape[1])
# y = y.reshape(y.shape[0], 1, y.shape[1])
# %%
import tensorflow.keras as keras
from tensorflow.keras.layers import Bidirectional, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam

input_layer = Input( shape=( 108, 4843))
lstm = Bidirectional(LSTM(200, activation="tanh", return_sequences=True, dropout=0.3))(input_layer, training = True)
lstm = Bidirectional(LSTM(200, activation="tanh",return_sequences=True, dropout=0.3))(lstm, training = True)
lstm = Bidirectional(LSTM(200, activation="tanh",return_sequences=True, dropout=0.3))(lstm, training = True)
lstm = LSTM(4843, activation = 'sigmoid', return_sequences=True, dropout=0.3)(lstm, training = True)
model = Model(input_layer, lstm)

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'accuracy'])
print(model.summary())


# input_layer = Input( shape=( 1, 4277))
# lstm = Bidirectional(LSTM(2138, activation="tanh", return_sequences=True, dropout=0.3))(input_layer, training = True)
# lstm = Bidirectional(LSTM(1000, activation="tanh",return_sequences=True, dropout=0.3))(lstm, training = True)
# lstm = Bidirectional(LSTM(200, activation="tanh",return_sequences=True, dropout=0.3))(lstm, training = True)
# lstm = LSTM(4277, activation = 'sigmoid', return_sequences=True, dropout=0.3)(lstm, training = True)
# model = Model(input_layer, lstm)

# model.compile(optimizer='adam', loss='mse', metrics=['mse', 'accuracy'])
# print(model.summary())
# from transformers import AutoTokenizer, AutoModelWithLMHead

# tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

# model = AutoModelWithLMHead.from_pretrained("gpt2-large").train()
# %%
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('lstm_model.hdf5', monitor='accuracy', verbose=1, save_best_only=True, mode='max')
batch_size=64
epochs=20

model.fit(x=oh_x_set, y=oh_y_set,
        batch_size=batch_size, epochs=epochs,
        verbose=1, callbacks=[checkpoint])

#%%

import numpy as np

preds = np.round(model.predict(oh_x_set[:10]))

# sents = vectorizer.inverse_transform(preds)

#%%
# from torch.utils import data
# import torch
# batch_size = 32

# x = torch.Tensor(x)
# y = torch.Tensor(y)
# dataset = data.TensorDataset(x, y)
# train_loader = data.DataLoader(dataset, batch_size=batch_size)
# #%%
# import torch.nn as nn

# learning_rate = 0.01
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# #%%

# class generator_model(nn.Module):
#     def __init__(self):
#         super(generator_model, self).__init__()
        
#         self.lstm1 = nn.LSTM(1077, 50, 4)

#         # self.dense3 = nn.Linear(in_features=64, out_features=10)
        
#     def forward(self, x):
        
#         x = nn.Softmax()(x)
#         return x
# # %%
# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.BCELoss()
# # %%

# from torch.utils import data

# datas = data.TensorDataset(x, y)
# dataloader = data.DataLoader(datas)
# # %%

# epochs= 10
# a = ''
# for e in range(epochs):
#     print(f"\nEpoch {e}")

#     for i, (q, r) in enumerate(dataloader):
#         print(f"\tbatch {i} / {len(dataloader)}", end=" ")

#         optimizer.zero_grad()

#         q = q.to(device).to(torch.long)
#         r = r.to(device).to(torch.long)
#         a = q

#         output = model(q)
        
#         loss = criterion(output, l)

#         loss.backward()
#         optimizer.step()
        
#         print(f" Loss: {loss.item()}")

#         torch.save(model.state_dict(), "models/gpt_retrained.ckpt")

# %%
