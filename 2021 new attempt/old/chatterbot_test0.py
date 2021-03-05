#%%
import pandas as pd
import numpy as np
from chatterbot import ChatBot

#%%

bot = ChatBot(
    'Buddy',  
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.TimeLogicAdapter'],
)

#%%
# Inport ListTrainer
from chatterbot.trainers import ListTrainer

trainer = ListTrainer(bot)

trainer.train([
'Hi',
'Hello',
'I need your assistance regarding my order',
'Please, Provide me with your order id',
'I have a complaint.',
'Please elaborate, your concern',
'How long it will take to receive an order ?',
'An order takes 3-5 Business days to get delivered.',
'Okay Thanks',
'No Problem! Have a Good Day!'
])

#%%
response = bot.get_response('I have a problem.')

print("Bot Response:", response)
# %%
name=input("Enter Your Name: ")
print("Welcome to the Bot Service! Let me know how can I help you?")
while True:
    request=input(name+':')
    if request=='Bye' or request =='bye':
        print('Bot: Bye')
        break
    else:
        response=bot.get_response(request)
        print('Bot:',response)
# %%
import spacy


#%%
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

print(f"Text, Lemma, POS, TAG, DEP, shape, is_alpha, is_stop")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
#%%
doc = nlp("I want to order a pizza")

print(f"Text, Lemma, POS, TAG, DEP, shape, is_alpha, is_stop")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
#%%
def arr_to_str(arr, delimiter=' '):
    big_str = ''

    for f in arr:
        big_str = big_str + f"{f}" + delimiter
    
    big_str = big_str[:-1]
    return big_str
#%%
reduced_sent = []
doc = nlp("I want to order a pizza")

print(f"Text, Lemma, POS, TAG, DEP, shape, is_alpha, is_stop")
for token in doc:
    if token.pos_ in ['VERB', 'NOUN']:
        reduced_sent.append(token.text)
arr_to_str(reduced_sent)
# %%
reduced_sent = []
doc = nlp("I want a large pizza with half olives and half onion")

print(f"Text, Lemma, POS, TAG, DEP, shape, is_alpha, is_stop")
for token in doc:
    # if token.pos_ in ['VERB', 'NOUN']:
    if token.is_stop:
        continue
    reduced_sent.append(token)
arr_to_str(reduced_sent)
# %%
