import os
from dotenv import main

from aiogram.dispatcher.filters import BoundFilter
from aiogram import types, Dispatcher


# CONST VARIABLES
class Config:
    main.load_dotenv()
    
    BOT_TOKEN: str = str(os.getenv('BOT_TOKEN'))
    ID_NTK_BIG_CHAT: int = -1001684546093
    ID_NTK_SMALL_CHAT: int = -1001384533622
    BLACK_LIST_OF_USERS = [2132881105, ]
    BAD_WORDS = []
    with open('bad_words.txt', 'r') as file:
        for line in file:
            word = line.strip()
            BAD_WORDS.append(word)
    try:
        DELTA_TIME_FOR_RECIEVE_NTK: int = int(os.getenv('DELTA_TIME'))
    except TypeError:
        DELTA_TIME_FOR_RECIEVE_NTK: int = 20

    try:
        SUPER_ADMINS: int = [int(id_admin) for id_admin in os.getenv('SUPER_ADMINS').split(',')]
    except TypeError:
        SUPER_ADMINS: list[int] = []
    

config = Config()


# FILTERS
class NtkGroup(BoundFilter):
    async def check(self, msg: types.Message) -> bool:
        return \
            msg.chat.id in [config.ID_NTK_BIG_CHAT, config.ID_NTK_BIG_CHAT] or \
            msg.from_user.id not in config.BLACK_LIST_OF_USERS or \
            msg.from_id.id in config.SUPER_ADMINS


class BlackList(BoundFilter):
    async def check(self, msg: types.Message) -> bool:
        return msg.from_user.id not in config.BLACK_LIST_OF_USERS
    

class SuperAdmins(BoundFilter):
    async def check(self, msg: types.Message) -> bool:
        return msg.from_user.id in config.SUPER_ADMINS
    

# init filters
def setup(dp: Dispatcher):
    dp.filters_factory.bind(NtkGroup)
    dp.filters_factory.bind(BlackList)
    dp.filters_factory.bind(SuperAdmins)