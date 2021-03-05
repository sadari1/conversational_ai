#%%
import pandas as pd
import numpy as np 

#%%
def arr_to_str(arr, delim = ' '):
    string = ''
    for f in arr:
        string = string + f'{f}' + delim
    string = string[:-1]
    return string

#%%
colors = ['red', 'green', 'blue']
clothes = ['shirt', 'dress', 'gown', 'jacket']
sleeves = ['short', 'medium', 'long', 'none']
brand = ['A', 'B', 'C']

product_column = ['p_id', 'brand', 'clothing', 'sleeve', 'color']

product_array = []
for f in range(1000):
    id_ = f
    brand_ = brand[np.random.randint(0, len(brand), 1)[0]]
    sleeve_ = sleeves[np.random.randint(0, len(sleeves), 1)[0]]
    clothes_ = clothes[np.random.randint(0, len(clothes), 1)[0]]
    colors_ = colors[np.random.randint(0, len(colors), 1)[0]]

    entry = [id_, brand_, clothes_, sleeve_, colors_]
    product_array.append(entry)

#%%
products = pd.DataFrame(product_array, columns=product_column)
products.to_csv("products.csv", index=False)
#%%
customer_column = ['c_id', 'p_id', 'date']

customer_array = []
for f in range(2000):
    c_id_ = np.random.randint(0, 51, 1)[0]
    p_id_ = np.random.randint(0, len(products), 1)[0]
    month = np.random.randint(1, 13, 1)[0]
    day = np.random.randint(1, 31, 1)[0]

    entry = [c_id_, p_id_, f"{month}-{day}-2020"]
    customer_array.append(entry)
#%%
customers = pd.DataFrame(customer_array, columns=customer_column)
customers.to_csv('customers.csv', index=False)

#%%
first_names = ['John', 'Mary', 'Sue', 'Bob', 'Jane', 'Richad', 'Kevin', 'Joe', 'Chad', 'Alex', 'Susan',
'Jennifer', 'Tiffany', 'Roger']
last_names = ['Smith', 'Watson', 'White', 'Johnson', 'Williams', 'Jones', 'Miller', 'Davis', 'Garcia',
'Thomas', 'Thompson', 'Haris']

customer_map_cols = ['c_id', 'first', 'last']
customer_map_arr = []
finished_names = []
for c_id in customers['c_id'].unique():
    while True:
        first = first_names[np.random.randint(0, len(first_names), 1)[0]]
        last = last_names[np.random.randint(0, len(last_names), 1)[0]]
        if f"{first} {last}" not in finished_names:
            break
    finished_names.append(f"{first} {last}")
    entry = [c_id, first, last]
    customer_map_arr.append(entry)

#%%
customer_map = pd.DataFrame(customer_map_arr, columns=customer_map_cols)
customer_map.to_csv('customer_map.csv', index=False)


#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%





































# # Construct Product DB

# product_columns = ['p_id', 'name', 'size', 'type', 'toppings', 'num_toppings', 'price']

# product_array = [[1, 'custom', 'large', 'HT', '', 0, 10],
# [2, 'custom', 'medium', 'HT', '', 0, 8],
# [3, 'Veggie Lover', 'large', 'HT', 'onion, tomato, green pepper', 3, 12],
# [4, 'Meat Lover', 'large', 'HT', 'pepperoni, sausage, bacon', 3, 12],
# [5, 'Chicago', 'large', 'D', '', 0, 11],
# [6, 'Mixed', 'large', 'HT', 'onion, pepperoni, tomato', 0, 10],
# [7, 'Breakfast', 'large', 'HT', 'sausage, bacon, tomato', 0, 10],
# [8, 'Student', 'medium', 'HT', 'onion, pepperoni, tomato', 0, 10],
# [9, 'Hawaiian', 'large', 'HT', 'pineapple, bacon, tomato', 0, 10],]

# products = pd.DataFrame(product_array, columns = product_columns)
# products.to_csv('products.csv', index=False)
# products.head()
# #%%
# # Construct Customer DB

# customer_columns = ['c_id', 'name', 'size', 'type', 'toppings', 'num_toppings', 'price']

# customer_array = [[1, 'custom', 'large', 'HT', 'onion', 1, 11],
# [1, 'custom', 'large', 'HT', 'onion', 1, 11],
# [2, 'custom', 'large', 'HT', 'onion', 1, 11],
# [1, 'custom', 'large', 'HT', 'onion', 1, 11],
# [3, 'custom', 'large', 'HT', 'onion', 1, 11],
# [1, 'custom', 'large', 'HT', 'onion', 1, 11]]

# name = ['custom','custom','custom','custom','custom','custom','custom','custom','custom','custom','custom','custom',
# 'custom','custom','custom','custom','custom','custom','custom','custom',
#  'Veggie Lover','Veggie Lover', 'Meat Lover','Meat Lover','Meat Lover', 'Chicago', 'Chicago', 'Chicago', 'Chicago', 
#  'Mixed', 'Breakfast', 'Student', 'Hawaiian']
# sizes = ['medium', 'large']
# type_ = ['HT', 'HT','HT','D']
# toppings = ['onion', 'green pepper', 'tomato', 'peppeoni', 'sausage', 'pineapple', 'bacon']
# #%%

# for f in range(1000):
#     customer_id = np.random.randint(1, 50, 1)[0]
#     name_ = np.random.randint(0, len(name), 1)[0]
#     name_ = name[name_]
#     if name_ is 'custom':
#         size_ = np.random.randint(0, 1, 1)[0]
#         size_ = sizes[size_]

#         pizza_type = np.random.randint(0, len(type_), 1)[0]
#         pizza_type = type_[pizza_type]

#         topping_list = np.random.randint(0, len(toppings), np.random.randint(0, 4, 1)[0])
#         num_toppings_ = len(topping_list)
#         topping_list = np.array(toppings)[topping_list]

#         topping_list = arr_to_str(topping_list, ', ')

#         price_ = products[(products['name'] == 'custom') & (products['size'] == size_)].iloc[0]['price'] + num_toppings_
#         entry = [customer_id, name_, size_, pizza_type, topping_list, num_toppings_, price_]
#     else:
#         entry = [customer_id] + list(products[products['name'] == name_].iloc[0])[1:]
#     customer_array.append(entry)
# #%%
# customers = pd.DataFrame(customer_array, columns = customer_columns)
# customers.to_csv('customers.csv', index=False)
# customers.head()
# #%%
# # Construct transaction history (customer's individual databank)
# #%%
# create a df that maps names to customer ID so that when user enters information to chatbot we can map this
# and try to make better predictions
# first_names = []
# last_names = []




#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

