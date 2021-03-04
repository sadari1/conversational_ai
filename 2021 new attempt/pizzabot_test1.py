#%%
import pandas as pd
import numpy as np
from chatterbot import ChatBot
#%%

edf_columns = ['name', '']

#%%

bot = ChatBot(
    'Pizzabot1',  
    logic_adapters=[
        'chatterbot.logic.BestMatch'
        # 'chatterbot.logic.TimeLogicAdapter'
        ], 
    read_only=True
)
#%%
order_response = ["Ok, let's order a pizza."]

cancel_response = ['Ok, first I need some information about the order.']

complaint_response = [
"I'm sorry to hear that!",
"We are sorry about that!",
"We apologize for your experience!"]

praise_responses = ["That's great to hear!",
"That's great news!",
"Thank you for your response!",
"We are glad to hear that!"
]
#%%
def submit_order(order):
    print(order)

#%%
def order_handler(name):
    order = {}
    order_correct = False
    while not order_correct:
        print("If you'd like to cancel the ordering process, simply type 'cancel'")
        size = input("Enter pizza size (Medium or Large)")

        if size[0].lower() == 'c':
            break

        toppings = input("What toppings would you like (onion, green pepper, tomato, pepperoni)")
        
        if toppings[0].lower() == 'c':
            break

        print(f"You have chosen a {size} pizza with the toppings: {toppings}")
        check_correct = input("Is this correct?")

        first_val = check_correct[0].lower() 

        if first_val == 'c':
            break

        while first_val not in ['y', 'n']:
            print("Sorry, I didn't understand. Enter 'Yes' or 'No'")
            check_correct = input("Is this correct?")
            first_val = check_correct[0].lower() 
        
        if first_val == 'y':
            print("Ok, submitting order. Thank you!")

            order['name'] = name
            order['size'] = size
            order['toppings'] = toppings

            submit_order(order)

            order_correct = True
        else:
            print("Ok, let's redo the order")
    
    
#%%
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
        
        if str(response) in order_response:
            order_handler(name)
            continue_sess = input("Thank you for ordering. Is there anything else we can help you with?")
            first_val = continue_sess[0].lower() 
        
            if first_val == 'y':
                continue
            else:
                print("Thank you for ordering, goodbye!")
                break


        if str(response) in cancel_response:
            print('cancel')
        if str(response) in complaint_response:
            print('complaint')
        if str(response) in praise_responses:
            print('praise')


# %%
