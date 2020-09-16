# # To add a new cell, type '# %%'
# # To add a new markdown cell, type '# %% [markdown]'
# # %%
# from IPython import get_ipython


# # %%
# get_ipython().system('python -m spacy link en_core_web_sm en')


# # %%
# import spacy
# spacy.__version__


# %%
from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot

bot = ChatBot(
    'Norman',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.sqlite3',
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand.',
            'maximum_similarity_threshold': 0.75
        },

        {
            'import_path': 'custom_adapter.MyLogicAdapter'
        }
    ]
)


# %%
from chatterbot.trainers import ListTrainer

trainer = ListTrainer(bot)

# trainer.train([
#     'How are you?',
#     'I am good.',
#     'That is good to hear.',
#     'Thank you',
#     'You are welcome.',
# ])

trainer.train([
    "How are you?",
    "I am good.",
    "That is good to hear.",
    "Thank you",
    "You are welcome.",
])


# %%
bot.get_response("That is good to hear.")


# %%
from chatterbot.logic import LogicAdapter


class MyLogicAdapter(LogicAdapter):

    def __init__(self, chatbot, **kwargs):
        super().__init__(chatbot, **kwargs)

        def can_process(self, statement):
            return True

        def process(self, input_statement, additional_response_selection_parameters):
            import random

            # Randomly select a confidence between 0 and 1
            confidence = random.uniform(0, 1)

            # For this example, we will just return the input as output
            # selected_statement = input_statement
            # selected_statement.confidence = confidence
            selected_statement = "None" 
            
            if input_statement == "asdf":
                selected_statement = "logic adapter"

            return selected_statement


