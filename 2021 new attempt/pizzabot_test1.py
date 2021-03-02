#%%
import pandas as pd
import numpy as np
from chatterbot import ChatBot


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

