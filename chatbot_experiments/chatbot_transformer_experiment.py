#%%
import pandas as pd

import numpy as np
import time 

import tensorflow as tf
import tensorflow_datasets as tfds
from nltk.corpus import stopwords
import nltk
words = set(nltk.corpus.words.words())

from transformer_redone import *

punctuation_list = ['.', ':', '!', '?', ',', '^', '(', ')', '。', '、', "'", ":/", '-', '/', '&', ';', '$', '*', '+', '\\', '_', '`', '"', '=']
#%%
df = pd.read_csv("../data/customer_service/twcs/twsc_sorted.csv")#, parse_dates=[3])


#%%

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
responses = responses.iloc[:]
in_resp_to_list = list(responses.in_response_to_tweet_id)

queries = df[df.tweet_id.isin(in_resp_to_list)]
query_id_list = list(queries.tweet_id)

responses = responses[responses.in_response_to_tweet_id.isin(query_id_list)]
responses = responses.drop_duplicates("in_response_to_tweet_id")

q = queries.text.apply(lambda row: row.encode('ascii',errors='ignore').decode()).apply(purge_at)
r = responses.text.apply(lambda row: row.encode('ascii',errors='ignore').decode())

toc = time.time()
print(f"Job took {toc-tic} seconds")
#%%

# Convert all to lower

lower_mapper = lambda x: x.lower()

q = q.apply(lower_mapper)
r = r.apply(lower_mapper)
#%%
# Get rid of http

extra_words = ['http', '@', '#', *list(range(10)), *punctuation_list, '[', ']']
def no_extra_words(text):
    sent = []
    text = text.split(" ")
    for word in text:
        check=True

        for purge in extra_words:
            if str(purge) in word:
                check = False
        if check:
            sent.append(word)
    return arr_to_str(sent)

tic = time.time()

q = q.apply(no_extra_words)#list(map(no_extra_words, q))
r = r.apply(no_extra_words)#list(map(no_extra_words, r))

toc = time.time()
print(f"Job took {toc-tic} seconds.")
#%%
def purge_non_english(text):
    # try:
    text = [w for w in nltk.wordpunct_tokenize(text) \
        if w.lower() in words or not w.isalpha()]
    # except:
    #     # print(text)

    #     try:
    #         print([w for w in nltk.wordpunct_tokenize(text) \
    #             if w.lower() in words or not w.isalpha()])
    #     except: 
    #         print("FAIL", text)
    #         print([w for w in nltk.wordpunct_tokenize(text) \
    #             if w.lower() in words or not w.isalpha()])
    return arr_to_str(text)

tic = time.time()

q = q.apply(purge_non_english)#list(map(purge_non_english, q))
r = r.apply(purge_non_english)#list(map(purge_non_english, r))


toc = time.time()
print(f"Job took {toc-tic} seconds.")
# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8

# input_vocab_size = tokenizer_pt.vocab_size + 2
# target_vocab_size = tokenizer_en.vocab_size + 2
# dropout_rate = 0.1

# %%

tic = time.time()

tokenizer_q = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (query for query in q), target_vocab_size=2**15)

tokenizer_r = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (resp for resp in r), target_vocab_size=2**15)

toc = time.time()

print(f"Job took {toc-tic} seconds.")
#%%

sample_string = 'Hello world!'

tokenized_string = tokenizer_q.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_q.decode(tokenized_string)
print ('The original string: {}'.format(original_string))



#%%
for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer_q.decode([ts])))

#%%
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def encode(lang1, lang2):
  lang1 = [tokenizer_q.vocab_size] + tokenizer_q.encode(
      lang1.numpy()) + [tokenizer_q.vocab_size+1]

  lang2 = [tokenizer_r.vocab_size] + tokenizer_r.encode(
      lang2.numpy()) + [tokenizer_r.vocab_size+1]

  return lang1, lang2

#%%
def tf_encode(q, r):
  result_q, result_r = tf.py_function(encode, [q, r], [tf.int64, tf.int64])
  result_q.set_shape([None])
  result_r.set_shape([None])

  return result_q, result_r

#%%
MAX_LENGTH = 400

tic = time.time()
def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

train_examples = tf.data.Dataset.from_tensor_slices((q, r))
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
toc=time.time()

print(f"Job took {toc-tic} seconds.")
#%%
# pt_batch, en_batch = next(iter(train_dataset))
# pt_batch, en_batch

#%%
num_layers = 6
d_model = 512
dff = 1024
num_heads = 8

input_vocab_size = tokenizer_q.vocab_size + 2
target_vocab_size = tokenizer_r.vocab_size + 2
dropout_rate = 0.1

#%%
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)



#%%
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

#%%
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


#%%

checkpoint_path = "./checkpoints/train/full2/"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')



#%%
EPOCHS = 50

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(tar_real, predictions)

#%%
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_dataset):
    # print(tokenizer_q.decode(inp[0]), tokenizer_r.decode(tar[0]))
    train_step(inp, tar)

    if batch % 1 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

#%%
def evaluate(inp_sentence):
  start_token = [tokenizer_q.vocab_size]
  end_token = [tokenizer_q.vocab_size + 1]

  # inp sentence is portuguese, hence adding the start and end token
  inp_sentence = start_token + tokenizer_q.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)

  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [tokenizer_r.vocab_size]
  output = tf.expand_dims(decoder_input, 0)

  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_r.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


#%%
def translate(sentence, plot=''):
  result, attention_weights = evaluate(sentence)

  predicted_sentence = tokenizer_r.decode([i for i in result 
                                            if i < tokenizer_r.vocab_size])  

  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(predicted_sentence))

  if plot:
    plot_attention_weights(attention_weights, sentence, result, plot)


#%%
translate("I've been waiting for my package for a while")
print ("Real translation: this is a problem we have to solve .")


# %%

for i in range(10):
  translate(queries.iloc[i].text)
  print(f"Real translation: {responses.iloc[i].text}\n")
# %%
