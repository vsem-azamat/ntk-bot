import os
from dotenv import main

from aiogram.dispatcher.filters import BoundFilter
from aiogram import types, Dispatcher


# CONST VARIABLES
class Config:
    main.load_dotenv()
    
    BOT_TOKEN: str = str(os.getenv('BOT_TOKEN'))
    DELTA_TIME_FOR_RECIEVE_NTK: int = int(os.getenv('DELTA_TIME'))
    ID_NTK_BIG_CHAT: int = -1001684546093
    ID_NTK_SMALL_CHAT: int = -1001384533622

    BLACK_LIST_OF_USERS = [2132881105, ]
    BAD_WORDS = []
    with open('bad_words.txt', 'r') as file:
        for line in file:
            word = line.strip()
            BAD_WORDS.append(word)

config = Config()

# FILTERS
class NtkGroup(BoundFilter):
    async def check(self, msg: types.Message) -> bool:
        return \
            msg.chat.id in [config.ID_NTK_BIG_CHAT, config.ID_NTK_BIG_CHAT] or \
            msg.from_user.id not in config.BLACK_LIST_OF_USERS


class BlackList(BoundFilter):
    async def check(self, msg: types.Message) -> bool:
        return msg.from_user.id not in config.BLACK_LIST_OF_USERS
    

# init filters
def setup(dp: Dispatcher):
    dp.filters_factory.bind(NtkGroup)
    dp.filters_factory.bind(BlackList)