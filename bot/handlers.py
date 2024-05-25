import io
import random

from aiogram.filters import Command
from aiogram import Bot, Router, types
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import cnfg
from bot.filters import NTKChatFilter, SuperAdmins
from apps.parse_functions import get_ntk_quantity
from apps.plot_functions import plotGraph
from apps.predictModels import predictModels
from apps.weather_api import weatherAPI
from apps.gpt import get_gpt_response


router = Router()


@router.message(Command('anon'))
async def anon(message: types.Message, bot: Bot):
    """Send anon message"""
    text_head = "<b>💌Анон плс:</b>\n\n"
    text = message.text[6:].strip()
    if message.chat.id == message.from_user.id and text:
        await bot.send_message(
            chat_id=cnfg.ID_NTK_BIG_CHAT,
            text=text_head + text,
            parse_mode='HTML'
            )
    else:
        await message.delete()


@router.message(Command('ntk'), NTKChatFilter())
async def ntk(message: types.Message):
    """Send ntk quantity"""
    q = await get_ntk_quantity()
    text = f"📚<b>В NTK сейчас людей:</b> {q}"
    text += '\nДохуя крч.' if q >= 700 else ''
    text += "\n\n📣<a href='t.me/ntk_info'><b>NTK info</b></a>"
    await message.answer(
        text=text,
        parse_mode='HTML', 
        )
    await message.delete()


@router.message(Command('help'))
async def help(message: types.Message):
    print(message)
    """Send help message"""
    text = \
    "🤖<b>Хай, я создан для чата @chat_ntk!</b>\n\n"\
    "📋<b>Команды:</b>\n"\
    "/ntk - Показать кол-во людей в NTK\n"\
    "/graph - Показать график посещений NTK\n"
    builder = InlineKeyboardBuilder()
    builder.add(
        types.InlineKeyboardButton(text='📚NTK chat', url='https://t.me/chat_ntk'),
        types.InlineKeyboardButton(text='👨‍🎓Admin', url='t.me/vsem_azamat'),
        types.InlineKeyboardButton(text='🧑‍💻GitHub', url='github.com/vsem-azamat/ntk_bot/')
    )
    builder.adjust(1)
    await message.answer(
        text=text, 
        reply_markup=builder.as_markup(), 
        disable_web_page_preview=True,
        parse_mode='HTML'
        )


@router.message(Command('graph'), NTKChatFilter())
async def send_stats(message: types.Message, bot: Bot):
    """Send graph with NTK visits prediction and weather forecast """
    fig_visits, _ = await plotGraph.daily_graph_with_predictions()
    
    buffer_visits = io.BytesIO()
    fig_visits.savefig(buffer_visits, format='png')
    buffer_visits.seek(0)
    
    await bot.send_photo(
        chat_id=message.chat.id, 
        photo=types.BufferedInputFile(
            file=buffer_visits.read(),
            filename='visits.png',
            )
        )
    await message.delete()


@router.message(Command('weather'), SuperAdmins())
async def send_weather(message: types.Message, bot: Bot):
    """Send weather forecast"""
    fig_weather, _, _ = await weatherAPI.plot_daily_weather_forecast()
    
    buffer_weather = io.BytesIO()
    fig_weather.savefig(buffer_weather, format='png')
    buffer_weather.seek(0)
    
    await bot.send_photo(
        chat_id=message.chat.id, 
        photo=types.BufferedInputFile(
            file=buffer_weather.read(),
            filename='weather.png',
            )
        )
    await message.delete()


@router.message(Command('learn'), SuperAdmins())
async def learn_models(msg: types.Message):
    """Learn models"""
    await predictModels.learn_models()
    await msg.answer('Models learned!')
    await msg.delete()


@router.message(Command('data'), SuperAdmins())
async def send_data(msg: types.Message, bot: Bot):
    """Send ntk_data.txt with NTK visits"""
    with open('ntk_data.txt', 'rb') as file:
        input_file = types.BufferedInputFile(
            file=file.read(), 
            filename='ntk_data.txt'
            )
        await bot.send_document(msg.chat.id, input_file)


@router.message(NTKChatFilter())
async def gpt_bullying(message: types.Message):
    """Random GPT response"""
    text = message.text
    if text and random.random() < 0.00015:
        response = await get_gpt_response(text)
        if response:
            await message.reply(str(response))
