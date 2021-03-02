#%%
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
#%%

#%%

#%%

#%%

