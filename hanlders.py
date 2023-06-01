import io
from datetime import datetime

from aiogram import types
from aiogram.dispatcher.filters import Command

from bot import dp, bot
from configs import NtkGroup, SuperAdmins
from apps.parse_functions import get_ntk_quantity, get_bad_word
from apps.plot_functions import pltGraph
from apps.predictModels import predictModels


@dp.message_handler(Command("ntk", prefixes='!/'), NtkGroup())
async def ntk(msg: types.Message):
    """Send ntk quantity"""
    q = await get_ntk_quantity()
    text = f'📚В NTK сейчас людей: {q}'
    if int(q) >= 700:
        text += '\nДохуя крч.'
    await msg.answer(text)
    await msg.delete()


@dp.message_handler(Command('help', prefixes='!/'), NtkGroup())
async def ask_ntk(msg: types.Message):
    """Send help message"""
    text = \
    "🤖<b>Хай, я создан для чата @chat_ntk!</b>\n\n"\
    "📋<b>Команды:</b>\n"\
    "/ntk - Показать кол-во людей в NTK\n"\
    "/graph - Показать график посещений NTK\n"\
    "/badword - Нужно переслать на чужое сообщение, чтобы оскорбить.\n"
    keyboard = types.InlineKeyboardMarkup()
    button_chat = types.InlineKeyboardButton(text='📚NTK chat', url='https://t.me/chat_ntk')
    buttons = [
        types.InlineKeyboardButton(text='👨‍🎓Admin', url='t.me/vsem_azamat'),
        types.InlineKeyboardButton(text='🧑‍💻GitHub', url='github.com/vsem-azamat/ntk_bot/')
        ]
    keyboard.add(button_chat)
    keyboard.add(*buttons)
    await msg.answer(text, reply_markup=keyboard, disable_web_page_preview=True)
    await msg.delete()


@dp.message_handler(Command('graph', prefixes='!/'), NtkGroup())
async def send_stats(msg: types.Message):
    """Send graph with prediction"""
    fig_predict = await pltGraph.daily_graph_with_predictions()
    image_buffer = io.BytesIO()
    fig_predict.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    await bot.send_photo(
        chat_id=msg.chat.id,
        photo=types.InputFile(image_buffer),
        caption=str(datetime.now().strftime('%d-%m-%Y'))
    )
    await msg.delete()


@dp.message_handler(Command('learn', prefixes='!/'), SuperAdmins())
async def learn_models(msg: types.Message):
    """Learn models"""
    await predictModels.learn_models()
    await msg.answer('Models learned!')
    await msg.delete()


@dp.message_handler(Command('badword', prefixes='!/'), NtkGroup())
async def send_bad_word(msg: types.Message):
    if msg.reply_to_message:
        text = await get_bad_word()
        await msg.reply_to_message.reply(text)
    await msg.delete()