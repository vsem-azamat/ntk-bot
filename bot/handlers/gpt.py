import random
from aiogram import Router, types

from apps.gpt import get_gpt_response
from bot.filters import NTKChatFilter


router = Router()


@router.message(NTKChatFilter())
async def gpt_bullying(message: types.Message):
    """Random GPT response"""
    text = message.text
    if text and random.random() < 0.025:
        response = await get_gpt_response(text)
        if response:
            await message.reply(str(response))
