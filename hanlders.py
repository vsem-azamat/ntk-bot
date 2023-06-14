import io

from aiogram import types
from aiogram.dispatcher.filters import Command

from bot import dp, bot
from config import NtkGroup, SuperAdmins
from apps.parse_functions import get_ntk_quantity
from apps.plot_functions import plotGraph
from apps.predictModels import predictModels
from apps.weather_api import weatherAPI


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
    """Send graph with NTK visits prediction and weather forecast """
    fig_visits, _ = await plotGraph.daily_graph_with_predictions()
    
    buffer_visits = io.BytesIO()
    fig_visits.savefig(buffer_visits, format='png')
    buffer_visits.seek(0)
    
    await bot.send_photo(
        chat_id=msg.chat.id, 
        photo=buffer_visits
        )
    await msg.delete()


@dp.message_handler(Command('weather', prefixes='!/'), SuperAdmins())
async def send_weather(msg: types.Message):
    """Send weather forecast"""
    fig_weather, _, _ = await weatherAPI.plot_daily_weather_forecast()
    
    buffer_weather = io.BytesIO()
    fig_weather.savefig(buffer_weather, format='png')
    buffer_weather.seek(0)
    
    await bot.send_photo(
        chat_id=msg.chat.id, 
        photo=buffer_weather
        )
    await msg.delete()


@dp.message_handler(Command('learn', prefixes='!/'), SuperAdmins())
async def learn_models(msg: types.Message):
    """Learn models"""
    await predictModels.learn_models()
    await msg.answer('Models learned!')
    await msg.delete()

