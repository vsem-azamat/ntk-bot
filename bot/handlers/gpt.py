import random
from aiogram import Router, types, Bot
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import cnfg
from apps.gpt import get_gpt_response
from bot.filters import NTKChatFilter, SuperAdmins


router = Router()


@router.message(NTKChatFilter())
async def gpt_bullying(message: types.Message):
    """Random GPT response"""
    text = message.text
    if text and random.random() < cnfg.GPT_ANSWER_PROBABILITY:
        response = await get_gpt_response(text)
        if response:
            await message.reply(str(response))


@router.message(Command('gpt'), SuperAdmins())
async def change_gpt_answer_probability(message: types.Message):
    """Change GPT answer probability"""
    builder = InlineKeyboardBuilder()
    builder.add(
        types.InlineKeyboardButton(
            text='2.5%',
            callback_data='gpt_0.025'
        ),
        types.InlineKeyboardButton(
            text='5%',
            callback_data='gpt_0.05'
        ),
        types.InlineKeyboardButton(
            text='10%',
            callback_data='gpt_0.1'
        ),
        types.InlineKeyboardButton(
            text='15%',
            callback_data='gpt_0.15'
        ),
        types.InlineKeyboardButton(
            text='20%',
            callback_data='gpt_0.2'
        ),
        types.InlineKeyboardButton(
            text='disable',
            callback_data='gpt_disable'
        )
    )
    builder.adjust(2)
    await message.reply(
        text='Choose GPT answer probability:',
        reply_markup=builder.as_markup()
    )
    

@router.callback_query(lambda callback_query: callback_query.data.startswith('gpt_'), SuperAdmins())
async def set_gpt_answer_probability(callback_query: types.CallbackQuery, bot: Bot):
    """Set GPT answer probability"""
    probability = callback_query.data[4:]
    try:
        cnfg.GPT_ANSWER_PROBABILITY = float(probability)
        text_probability = f'{cnfg.GPT_ANSWER_PROBABILITY*100}%'
    except ValueError:
        cnfg.GPT_ANSWER_PROBABILITY = -1
        text_probability = 'disable'

    await callback_query.message.edit_text(
        text=f'GPT answer probability: {text_probability}',
        reply_markup=None
    )

