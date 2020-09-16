#%%

import pandas as pd
import numpy as np
import spacy
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from transformer import *


#%%
df = pd.read_csv("data/movie_sentiment/train.tsv", sep="\t", nrows=5000)

#%%
df


#%%
asdf = ""
embeddings_dict = {}
with open("glove/glove.840B.300d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()

        asdf = (line, values)
        word = values[0]
        vector = np.asarray(values[1:])#, "float32")
        embeddings_dict[word] = vector

#%%
embeddings_dict

# word embeddings that are length 300.

#%%

text_collection = np.array(df.Phrase)

#%%
stop_words = ['a', 'is', 'it', 'the', 'of', 'to', '.', ',']
# %%


#%%

def get_emb_vec(sentence):
  split = sentence.split(" ")
  temp_arr = []

  for word in split:
    if word not in stop_words:
      try:
        word_emb = embeddings_dict[word]
      except: 
        print("word not found: ", word)
        continue
      if word_emb.shape[0] == 300:
        temp_arr.append(word_emb)
  
  return np.array(temp_arr)

#%%

get_emb_vec(corpus[0])

#%%
text_body = np.array(df.Phrase)

corpus = []
for sent in text_body:
  transformed = get_emb_vec(sent)
  if transformed.shape[0] != 0:
    corpus.append(transformed)

corpus = np.array(corpus)

#%%
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(np.array(df.Phrase))

#%%
#%%


x_train = X.toarray()

y_train = tf.keras.utils.to_categorical(np.array(df.Sentiment))
#%%


    
#%%

vocab_size = len(vectorizer.vocabulary_)
#%%

num_layers = 2
d_model = 8
num_heads = 8
dff = 64
input_vocab_size = vocab_size
target_vocab_size = 5
pe_input = input_vocab_size
pe_target = target_vocab_size
rate = 0.1


transformer = Transformer(num_layers=num_layers,
                          d_model=d_model,
                          num_heads=num_heads,
                          dff=dff,
                          input_vocab_size=input_vocab_size,
                          target_vocab_size=target_vocab_size,
                          pe_input=pe_input,
                          pe_target=pe_target,
                          rate=rate)
#%%

learning_rate = CustomSchedule(512)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
#%%

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

#%%

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


#%%

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


#%%
checkpoint_path = "./checkpoints/sent_train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

#%%

EPOCHS = 128

#%%

# train_step_signature = [
#     # tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     # tf.TensorSpec(shape=(None, None), dtype=tf.int64),

#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# ]
# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
#   tar_inp = tar[:, :-1]
#   tar_real = tar[:, 1:]

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

BATCH_SIZE = 128
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)

#%%

for epoch in range(EPOCHS):
  start = time.time()
  
  train_loss.reset_states()
  train_accuracy.reset_states()
  
  # inp -> portuguese, tar -> english
  for batch, (inp, tar) in enumerate(train_loader):
    
    # inp = tf.convert_to_tensor(data[0], dtype=tf.int64)
    # tar = tf.convert_to_tensor(data[1], dtype=tf.int64)

    # inp = tf.TensorSpec.from_tensor(inp)
    # tar = tf.TensorSpec.from_tensor(tar)
    train_step(inp, tar)
    
    if batch % 50 == 0:
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
for batch, (inp, tar) in enumerate(train_loader):
    if batch == 0:
        evaluate(inp_sentence=inp)


#%%
test = ""

def evaluate(inp_sentence):
  test = inp_sentence
  vocab_size = len(vectorizer.vocabulary_)
  
  start_token = [vocab_size]
  end_token = [vocab_size + 1]
  
  # inp sentence is portuguese, hence adding the start and end token
  inp_sentence = start_token + vectorizer.transform(inp_sentence[0]) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [5]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(X.toarray().shape[1]):
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
    if predicted_id == vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

"""
num layers
d_model (256 for example)
num_heads 9num multi attention heads?)
dff
input_vocab size
target vocab size
pe_input = inputvocab size
pe_target = targetvocabsize
rate = dropoutrate, like 0.1.

"""

# %%

