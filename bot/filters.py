from typing import Union

from aiogram.types import Message
from aiogram.filters import BaseFilter

from config import cnfg


class ChatTypeFilter(BaseFilter):
    def __init__(self, chat_type: Union[str, list]):
        self.chat_type = chat_type

    async def __call__(self, message: Message) -> bool:
        if isinstance(self.chat_type, str):
            return message.chat.type == self.chat_type
        else:
            return message.chat.type in self.chat_type


class NTKChatFilter(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        return message.chat.id in [cnfg.ID_NTK_BIG_CHAT, cnfg.ID_NTK_SMALL_CHAT, cnfg.ID_NTK_CHANNEL]


class SuperAdmins(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        return message.from_user.id in cnfg.SUPER_ADMINS
