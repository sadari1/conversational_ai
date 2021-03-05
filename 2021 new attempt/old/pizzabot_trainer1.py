#%%
import pandas as pd
import numpy as np
from chatterbot import ChatBot


#%%
training_sents = np.load('training_sents.npy')
training_sents = [str(f) for f in training_sents]
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
# Inport ListTrainer
from chatterbot.trainers import ListTrainer

trainer = ListTrainer(bot)


# trainer.train('chatterbot.corpus.english')

#%%
for e in range(1):
    print(f"Epoch {e}")
    trainer.train(training_sents)

#%%

response = bot.get_response('Can I place an order for a pizza')

print("Bot Response:", response)
# %%
