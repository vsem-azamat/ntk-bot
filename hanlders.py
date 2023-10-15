import io
import asyncio

from aiogram import types
from aiogram.dispatcher.filters import Command

from bot import dp, bot
from config import NtkGroup, SuperAdmins, config
from apps.parse_functions import get_ntk_quantity
from apps.plot_functions import plotGraph
from apps.predictModels import predictModels
from apps.weather_api import weatherAPI


@dp.message_handler(Command("anon", prefixes='!/'), )
async def anon(msg: types.Message, command: types.BotCommand):
    """Send anon message"""
    text_head = "<b>💌Анон плс:</b>\n\n"
    text = command.__dict__.get('args', None)
    if msg.chat.id == msg.from_user.id and text:
        await bot.send_message(
            chat_id=config.ID_NTK_BIG_CHAT,
            text=text_head + text,
            disable_notification=True,
            parse_mode=types.ParseMode.HTML
            )
    else:
        await msg.delete()


@dp.message_handler(Command("ntk", prefixes='!/'), NtkGroup())
async def ntk(msg: types.Message):
    """Send ntk quantity"""
    q = await get_ntk_quantity()
    text = f"📚<b>В NTK сейчас людей:</b> {q}"
    text += '\nДохуя крч.' if q >= 700 else ''
    text += "\n\n📣<a href='t.me/ntk_info'><b>NTK info</b></a>"
    await msg.answer(text)
    await msg.delete()


@dp.message_handler(Command('help', prefixes='!/'), NtkGroup())
async def ask_ntk(msg: types.Message):
    """Send help message"""
    text = \
    "🤖<b>Хай, я создан для чата @chat_ntk!</b>\n\n"\
    "📋<b>Команды:</b>\n"\
    "/ntk - Показать кол-во людей в NTK\n"\
    "/graph - Показать график посещений NTK\n"
    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(
        types.InlineKeyboardButton(text='📚NTK chat', url='https://t.me/chat_ntk'),        
    )
    keyboard.add(
        types.InlineKeyboardButton(text='📣NTK info', url='https://t.me/ntk_info')
    )
    keyboard.add(
        types.InlineKeyboardButton(text='👨‍🎓Admin', url='t.me/vsem_azamat'),
        types.InlineKeyboardButton(text='🧑‍💻GitHub', url='github.com/vsem-azamat/ntk_bot/')
    )
    message = await msg.answer(text, reply_markup=keyboard, disable_web_page_preview=True)
    await msg.delete()
    await asyncio.sleep(120)
    await message.delete()


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


@dp.message_handler(Command('data', prefixes='!/'), SuperAdmins())
async def send_data(msg: types.Message):
    with open('ntk_data.txt', 'rb') as file:
        await bot.send_document(msg.chat.id, file)