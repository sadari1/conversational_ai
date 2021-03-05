#%%
import numpy as np 
import spacy


#%%
nlp = spacy.load("en_core_web_sm")
#%%

order_response = ["Ok, let's order a pizza."]
order_utterances = [
'I want to order a pizza',
'I want to place an order',
'Can I place an order?',
'Can I order a pizza?',
'I want a pizza',
'Can I get a pizza?'
]

cancel_response = ['Ok, first I need some information about the order.']
cancel_utterances = [
'Can I cancel my order?',
'I want to cancel my order',
"I'd like to cancel an order",
"I'm cancelling my order"
]

complaint_response = [
"I'm sorry to hear that!",
"We are sorry about that!",
"We apologize for your experience!"]

complaint_utterances = [
'My pizza has not arrived yet',
"Where is my pizza? I ordered it an hour ago",
"Why did my pizza not come yet",
"Why is the delivery taking so long",
"I ordered a pizza and it still has not arrived",

"My pizza is the wrong type",
"This isn't the pizza I ordered",
"This pizza has the wrong toppings",
"I did not order this pizza",
"This pizza is the wrong size",
"You delivered the wrong pizza",

"The pizza arrived cold",
"The pizza tasted bad",
"I didn't like the pizza",
"The pizza did not taste good",
"The pizza was stale",

'I have a complaint about my order',
"There's a problem with my pizza",
"I want to make a complaint"
]

praise_responses = ["That's great to hear!",
"That's great news!",
"Thank you for your response!",
"We are glad to hear that!"
]
praise_utterances = [
"I was very happy with the service",
"The pizza was hot and it came on time",
"I loved the taste of the pizza",
"The pizza was great",
"I liked the service",
"The pizza was delivered quickly"
]
#%%
def arr_to_str(arr, delimiter=' '):
    big_str = ''

    for f in arr:
        big_str = big_str + f"{f}" + delimiter
    
    big_str = big_str[:-1]
    return big_str

#%%
full_dataset = []

for sent in order_utterances:
    # reduced_sent = []
    # doc = nlp(sent)

    # # print(f"Text, Lemma, POS, TAG, DEP, shape, is_alpha, is_stop")
    # for token in doc:
    #     # if token.pos_ in ['VERB', 'NOUN']:
    #     if token.is_stop:
    #         continue
    #     reduced_sent.append(token)
    # sent = arr_to_str(reduced_sent)
    for resp in order_response:
        full_dataset.append(sent)
        full_dataset.append(order_response)

for sent in cancel_utterances:
    # reduced_sent = []
    # doc = nlp(sent)

    # # print(f"Text, Lemma, POS, TAG, DEP, shape, is_alpha, is_stop")
    # for token in doc:
    #     # if token.pos_ in ['VERB', 'NOUN']:
    #     if token.is_stop:
    #         continue
    #     reduced_sent.append(token)
    # sent = arr_to_str(reduced_sent)
    for resp in cancel_response:
        full_dataset.append(sent)
        full_dataset.append(cancel_response)

for sent in complaint_utterances:
    # reduced_sent = []
    # doc = nlp(sent)

    # # print(f"Text, Lemma, POS, TAG, DEP, shape, is_alpha, is_stop")
    # for token in doc:
    #     # if token.pos_ in ['VERB', 'NOUN']:
    #     if token.is_stop:
    #         continue
    #     reduced_sent.append(token)
    # sent = arr_to_str(reduced_sent)
    for resp in complaint_response:
        full_dataset.append(sent)
        full_dataset.append(resp)

for sent in praise_utterances:
    # reduced_sent = []
    # doc = nlp(sent)

    # # print(f"Text, Lemma, POS, TAG, DEP, shape, is_alpha, is_stop")
    # for token in doc:
    #     # if token.pos_ in ['VERB', 'NOUN']:
    #     if token.is_stop:
    #         continue
    #     reduced_sent.append(token)
    # sent = arr_to_str(reduced_sent)
    for resp in praise_responses:
        full_dataset.append(sent)
        full_dataset.append(resp)


# %%
np.save("training_sents.npy", full_dataset)
# %%
