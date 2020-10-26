#%%
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, TweetTokenizer

import string
import time
from nltk.corpus import stopwords

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam

punctuation_list = ['.', ':', '!', '?', ',', '^']
# %%
df = pd.read_csv("../data/customer_service/twcs/twsc_sorted.csv")#, parse_dates=[3])

# %%

'''
Will try not cleaning punctuation to see how that affects anything.

'''
tic = 0
toc = 0
def timer(func_, param_list=[]):
    tic = time.time()

    output = func_(*param_list)
    toc = time.time()
    print(f"Job took {toc-tic} seconds.")

    return output

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

def start_time():
    tic = time.time()

def stop_time():
    toc = time.time()
    print(f"Job took {toc-tic} seconds.")
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

q = queries.text.apply(purge_at)
r = responses.text

toc = time.time()
print(f"Job took {toc-tic} seconds")
#%%

# Convert all to lower

lower_mapper = lambda x: x.lower()

q = q.apply(lower_mapper)
r = r.apply(lower_mapper)

# %%
tweet_tokenizer = TweetTokenizer()

tic = time.time()

q = q.apply(tweet_tokenizer.tokenize)
r = r.apply(tweet_tokenizer.tokenize)
# eng_sents = list(map(tweet_tokenizer.tokenize, eng_sents))
# fra_sents = list(map(tweet_tokenizer.tokenize, fra_sents))

toc = time.time()

print(f"Job took {toc-tic} seconds.")

#%%

def purge_stopwords(text):
    text = set(text)
    to_purge = set(stopwords.words('english') + punctuation_list)

    return list(text.difference(to_purge))


def _(eng_sents):
    eng_sents = eng_sents.apply(purge_stopwords)
    return eng_sents

q = timer(_, [q])
r = timer(_, [r])

#%%
# Get rid of http

extra_words = ['http', '@', '#']
def no_extra_words(text):
    sent = []

    for word in text:
        check=True

        for purge in extra_words:
            if purge in word:
                check = False
        if check:
            sent.append(word)
    return sent


q = list(map(no_extra_words, q))
r = list(map(no_extra_words, r))

# %%

def build_vocabulary(text):
    word_list = np.concatenate([line for line in text])

    unique_words = np.unique(word_list)

    vocab = dict(zip(unique_words, range(4, len(unique_words)+ 4)))

    vocab['<SOS>'] = 0
    vocab['<EOS>'] = 1
    vocab['<UNK>'] = 2
    vocab['<PAD>'] = 3

    return vocab

vocab = timer(build_vocabulary, [list(q) + list(r)])

# reverse_eng_vocab = dict(zip(eng_vocab.values(), eng_vocab.keys()))
# reverse_fra_vocab = dict(zip(fra_vocab.values(), fra_vocab.keys()))

#%%

q_max = np.max([len(text) for text in q])
r_max = np.max([len(text) for text in r])

#%%

# Pad ends of sentences until max length is reached.

def pad_sentences(text, max_len):
    num_to_pad = max_len - len(text)

    text = text + ["<PAD>" for f in range(num_to_pad)]
    return text

tic = time.time()

q = list(map(pad_sentences, q, q_max*np.ones(len(q), dtype=np.int32)))
r = list(map(pad_sentences, r, r_max*np.ones(len(r), dtype=np.int32)))

toc = time.time()
print(f"Job took {toc-tic} seconds.")
# %%

# def int_sequence(text, vocab):
#     return list(map(vocab.get, text))


def append_sent_tokens(sent):
    sent = ['<SOS>', *sent, '<EOS>']
    return sent


tic = time.time()

q = list(map(append_sent_tokens, q))
r = list(map(append_sent_tokens, r))

toc = time.time()
print(f"Job took {toc-tic} seconds.")

# %%

def tokenize_sents(sent, vocab):
    return list(map(vocab.get, sent))
# %%

tic = time.time()

q_tokenized = list(map(tokenize_sents, q, [vocab for f in range(len(q))]))
r_tokenized = list(map(tokenize_sents, r, [vocab for f in range(len(r))]))

toc = time.time()
print(f"Job took {toc-tic} seconds.")

#%%

q = np.array(q_tokenized)
r = np.array(r_tokenized)

np.save("q.npy", q)
np.save("r.npy", r)

#%%
num_encoder_tokens = q.shape[1]
num_decoder_tokens = r.shape[1]

latent_dim = 1000

# %%

encoder_inputs = Input(shape=(None,))
enc_x = Embedding(len(vocab), latent_dim,)(encoder_inputs)
enc_lstm = LSTM(units = latent_dim,
                    activation = "relu",
                    return_sequences = False,                    
                           return_state=True)
x, state_h, state_c = enc_lstm(enc_x)

encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, 1))
# dec_x= Embedding(len(fra_vocab), latent_dim)(decoder_inputs)
decoder_lstm= LSTM(units = latent_dim,
                    activation = "relu",
                    return_sequences = True,
                    return_state = True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(len(vocab), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# # Define an input sequence and process it.
# encoder_inputs = Input(shape=(None, num_encoder_tokens))
# encoder = LSTM(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# # We discard `encoder_outputs` and only keep the states.
# encoder_states = [state_h, state_c]

# # Set up the decoder, using `encoder_states` as initial state.
# decoder_inputs = Input(shape=(None, num_decoder_tokens))
# # We set up our decoder to return full output sequences,
# # and to return internal states as well. We don't use the 
# # return states in the training model, but we will use them in inference.
# decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                      initial_state=encoder_states)
# decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)

# # Define the model that will turn
# # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print(model.summary())
# %%

# Compile & run training
model.compile(optimizer=Adam(lr = 0.002), loss=sparse_categorical_crossentropy, metrics = ['accuracy'])
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

#%%
# encoder_eng_input = eng.reshape((-1, eng_max+2))
decoder_r_input = r.reshape((-1, r_max+2, 1))[:, :-1, :]
decoder_r_target = r.reshape((-1, r_max+2, 1))[:, 1:, :]

# %%
batch_size = 1024
epochs = 100
# Train model as previously
model.fit([q, decoder_r_input], decoder_r_target,
          batch_size=batch_size,
          epochs=epochs)


# %%
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_inputs = Input(shape=(None,))
# x = Embedding(len(fra_vocab), latent_dim)(decoder_inputs)
# decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

#%%

reverse_dict = dict(zip(vocab.values(), vocab.keys()))
max_decoder_seq_length = 200

# %%
def decode_sequence(input_seq):

    initial_states = encoder_model.predict(input_seq)
    # Initialize decoder input as a length 1 sentence containing "startofsentence",
    # --> feeding the start token as the first predicted word
    prev_word = np.zeros((1,1,1))
    prev_word[0, 0, 0] = vocab["<SOS>"]

    stop_condition = False
    translation = []
    while not stop_condition:
        # 1. predict the next word using decoder model
        logits, last_h, last_c = decoder_model.predict([prev_word] + initial_states)
        
        # 2. Update prev_word with the predicted word
        predicted_id = np.argmax(logits[0, 0, :])
        if predicted_id == 0:
            predicted_id = 1
        predicted_word = reverse_dict[predicted_id]
        translation.append(predicted_word)

        # 3. Enable End Condition: (1) if predicted word is "endofsentence" OR
        #                          (2) if translated sentence reached maximum sentence length
        if (predicted_word == '<EOS>' or len(translation) > r_max+2):
            stop_condition = True

        # 4. Update prev_word with the predicted word
        prev_word[0, 0, 0] = predicted_id

        # 5. Update initial_states with the previously predicted word's encoder output
        initial_states = [last_h, last_c]

    return " ".join(translation).replace('<EOS>', '')

#%%

def process_input_text(text):
    lower_mapper = lambda x: x.lower()
    text = list(map(lower_mapper, text))
    text = list(map(tweet_tokenizer.tokenize, text))
    text = list(map(purge_stopwords, text))
    text = list(map(pad_sentences, text, q_max*np.ones(len(q), dtype=np.int32)))
    text = list(map(append_sent_tokens, text))
    text = list(map(tokenize_sents, text, [vocab for f in range(len(text))]))

    return text


# %%

index = 29
sample = responses[index:index+1].text
print(sample)

true = queries[index:index+1].text
print(true)


sample = process_input_text(sample)

sample = np.array(sample).reshape((1, q_max+2))#[:1]

decode_sequence(sample.astype(np.float32))

# %%

for i in range(0, 100):
    sample = responses[i:i+1].text

    true = queries[i:i+1].text

    print(f"Query: {true}\Response: {sample}")
    sample = process_input_text(sample)

    sample = np.array(sample).reshape((1, q_max+2)).astype(np.float32)#[:1]
    decoded = decode_sequence(sample).replace("<PAD>", "")
    print(f'\nTranslated: {decoded}\n')
# %%

# %%
# Encode the input as state vectors.
states_value = encoder_model.predict(sample)

# Generate empty target sequence of length 1.
target_seq = np.zeros(( 1,1))
# Populate the first character of target sequence with the start character.
target_seq[0,0] = eng_vocab['<SOS>']

# Sampling loop for a batch of sequences
# (to simplify, here we assume a batch of size 1).
stop_condition = False
decoded_sentence = ''
while not stop_condition:
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # [target_seq] + states_value)

    # Sample a token
    sampled_token_index = np.argmax(output_tokens[0, 0, :])
    sampled_char = reverse_fra_dict[sampled_token_index]
    decoded_sentence += " "+ sampled_char

    # Exit condition: either hit max length
    # or find stop character.
    if (sampled_char == '<EOS>' or
        len(decoded_sentence) > max_decoder_seq_length):
        stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1,num_decoder_tokens))
    target_seq[0,0] = sampled_token_index

    # Update states
    states_value = [h, c]
decoded_sentence
# %%

eng_sentence = sample#eng_sentence.reshape((1, eng_max+2))  # give batch size of 1
initial_states = encoder_model.predict(eng_sentence)
# Initialize decoder input as a length 1 sentence containing "startofsentence",
# --> feeding the start token as the first predicted word
prev_word = np.zeros((1,1,1))
prev_word[0, 0, 0] = fra_vocab["<SOS>"]

stop_condition = False
translation = []
while not stop_condition:
    # 1. predict the next word using decoder model
    logits, last_h, last_c = decoder_model.predict([prev_word] + initial_states)
    
    # 2. Update prev_word with the predicted word
    predicted_id = np.argmax(logits[0, 0, :])
    if predicted_id == 0:
        predicted_id = 1
    predicted_word = reverse_fra_dict[predicted_id]
    translation.append(predicted_word)

    # 3. Enable End Condition: (1) if predicted word is "endofsentence" OR
    #                          (2) if translated sentence reached maximum sentence length
    if (predicted_word == 'endofsentence' or len(translation) > fra_max+2):
        stop_condition = True

    # 4. Update prev_word with the predicted word
    prev_word[0, 0, 0] = predicted_id

    # 5. Update initial_states with the previously predicted word's encoder output
    initial_states = [last_h, last_c]

" ".join(translation).replace('<EOS>', '')
# %%
