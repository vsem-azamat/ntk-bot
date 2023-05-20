import os
from dotenv import main

from aiogram.dispatcher.filters import BoundFilter
from aiogram import Dispatcher, types


# CONST VARIABLES
class Config:
    main.load_dotenv()
    
    BOT_TOKEN = str(os.getenv('BOT_TOKEN'))
    DELTA_TIME_FOR_RECIEVE_NTK = int(os.getenv('DELTA_TIME'))
    BAD_WORDS = []
    with open('bad_words.txt', 'r') as file:
        for line in file:
            word = line.strip()
            BAD_WORDS.append(word)
    ID_NTK_BIG_CHAT = -1001684546093
    ID_NTK_SMALL_CHAT = -1001384533622


# FILTERS
class NtkGroup(BoundFilter):
    async def check(self, message: types.Message) -> bool:
        return message.chat.id in [Config.ID_NTK_BIG_CHAT, Config.ID_NTK_BIG_CHAT]
