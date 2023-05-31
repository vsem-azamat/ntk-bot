import io
from datetime import datetime, timedelta

from aiogram import types
from aiogram.dispatcher.filters import Command

from bot import dp, bot
from configs import NtkGroup, SuperAdmins
from apps.parse_functions import get_ntk_quantity
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
    "<b>Хай, я создан для чата @chat_ntk!</b>\n\n"\
    "Команды:\n"\
    "/ntk - Показать кол-во людей в NTK\n\n"\
    "admin: t.me/vsem_azamat\n"\
    "GitHub: github.com/vsem-azamat/ntk_bot/"
    await msg.answer(text, disable_web_page_preview=True)
    await msg.delete()


@dp.message_handler(Command('graph', prefixes='!/'), SuperAdmins())
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


@dp.message_handler(Command('learn', prefixes='!/'), SuperAdmins())
async def learn_models(msg: types.Message):
    """Learn models"""
    await predictModels.learn_models()
    await msg.answer('Models learned!')
    await msg.delete()