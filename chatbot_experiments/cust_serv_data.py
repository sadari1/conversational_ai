#%%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#%%
df = pd.read_csv("../data/customer_service/twcs/twsc_sorted.csv", parse_dates=[3])
# df = pd.read_csv("data/customer_service/twcs/twsc_sorted.csv", parse_dates=[3])
# df = pd.read_csv("data/customer_service/twcs/twcs.csv", parse_dates=[3]).sort_values(by="created_at")
# df.to_csv("data/customer_service/twcs/twsc_sorted.csv", index=False)


#%%

amazon = df[df.author_id == 'AmazonHelp']


#%%



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

for f in range(responses.iloc[:100].shape[0]):

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

    
# %%

corpus = x + y 

vocab_list = []

for line in corpus:
    tokens = line.split(" ")
    for token in tokens:
        if token not in vocab_list:
            vocab_list.append(token)
    


#%%


# %%

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

input_layer = Input( shape=(96,  1077))
lstm = Bidirectional(LSTM(500, activation="tanh", return_sequences=True, dropout=0.3))(input_layer, training = True)
lstm = Bidirectional(LSTM(250, activation="tanh",return_sequences=True, dropout=0.3))(lstm, training = True)
lstm = Bidirectional(LSTM(538, activation="tanh",return_sequences=True, dropout=0.3))(lstm, training = True)
lstm = LSTM(1077, activation = 'sigmoid', return_sequences=True, dropout=0.3)(lstm, training = True)
model = Model(input_layer, lstm)

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'accuracy'])
print(model.summary())
# from transformers import AutoTokenizer, AutoModelWithLMHead

# tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

# model = AutoModelWithLMHead.from_pretrained("gpt2-large").train()
# %%
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('lstm_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
batch_size=64
epochs=1000

model.fit(x=x, y=y,
        batch_size=batch_size, epochs=epochs,
        verbose=1, callbacks=[checkpoint])

#%%

import numpy as np

preds = np.round(model.predict(x))

sents = vectorizer.inverse_transform(preds)

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
