import io
from aiogram import Bot, Router, types
from aiogram.filters import Command

from apps.plot_functions import plotGraph
from apps.predictModels import predictModels
from apps.weather_api import weatherAPI
from bot.filters import NTKChatFilter, SuperAdmins


router = Router()


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
