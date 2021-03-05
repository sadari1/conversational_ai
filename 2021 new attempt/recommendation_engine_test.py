#%%
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

#%%
products = pd.read_csv('products_w_soup.csv')
customers = pd.read_csv('customers.csv')
customer_map = pd.read_csv('customer_map.csv')

#%%
def arr_to_str(arr, delim = ' '):
    string = ''
    for f in arr:
        string = string + f'{f}' + delim
    string = string[:-1]
    return string
#%%
def get_recommendations(arg_dict):
    products = arg_dict['products']
    customers = arg_dict['customers']
    customer_map = arg_dict['customer_map']
    first = arg_dict['first']
    last = arg_dict['last']

    if first is not '' and last is not '':
            
        filt_ser = pd.Series([first, last], index=['first', 'last'])
        match = customer_map.loc[(customer_map[['first', 'last']] == filt_ser).all(axis=1)]
        c_id = match.iloc[0]['c_id']
        transaction_history = customers[customers['c_id'] == c_id]

        most_common_pid = transaction_history['p_id'].value_counts().index[0]
        most_common_product = products[products['p_id'] == most_common_pid].iloc[0]
        most_common_product = most_common_product['soup']

        embeddings1 = model.encode(most_common_product, convert_to_tensor=True)
        embeddings2 = model.encode(products['soup'], convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        cosine_scores = np.array(cosine_scores)

        products['scores'] = n_scores
        top_5 = products.sort_values('scores', ascending=False).drop_duplicates('soup').iloc[1:6]
        products = products.drop('scores', axis=1)
        return top_5
    else:
        top_5_pid = customers['p_id'].value_counts().index[:5]
        top_5 = products[products['p_id'].isin(top_5_pid)]
        if 'scores' in top_5.columns:
            top_5 = top_5.drop('scores', axis=1)
        return top_5
#%%
# def get_cid(customer_map, first, last):

# first = 'Bob'
# last = 'Smith'

first = ''
last = ''


arg_dict = {
    'products': products,
    'customers': customers,
    'customer_map': customer_map,
    'first': first,
    'last': last
}
get_recommendations(arg_dict)
#%%
filt_ser = pd.Series([first, last], index=['first', 'last'])
match = customer_map.loc[(customer_map[['first', 'last']] == filt_ser).all(axis=1)]
c_id = match.iloc[0]['c_id']
#%%
transaction_history = customers[customers['c_id'] == c_id]
#%%
most_common_pid = transaction_history['p_id'].value_counts().index[0]
#%%
most_common_product = products[products['p_id'] == most_common_pid].iloc[0]
most_common_product = most_common_product['soup']
most_common_product
#%%
# products['soup'] = products.apply(lambda x: arr_to_str(x[1:]), axis=1)
# products['soup']
#%%

#%%


embeddings1 = model.encode([most_common_product for f in range(len(products))], convert_to_tensor=True)
embeddings2 = model.encode(products['soup'], convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

#%%
n_scores = []
for f in range(len(products['soup'])):
    n_scores.append(cosine_scores[f][f].item())
#%%
products['scores'] = n_scores
#%%
products.sort_values('scores', ascending=False).drop_duplicates('soup')
#%%

#%%
embeddings1 = model.encode(most_common_product, convert_to_tensor=True)
embeddings2 = model.encode(products['soup'], convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
#%%

#%%

#%%

#%%

#%%

#%%

