import io

from aiogram import Bot, Router, types
from aiogram.filters import Command

from apps.plot_functions import plotGraph
from apps.predictModels import predictModels
from bot import db
from bot.filters import NTKChatFilter, SuperAdmins

router = Router()


@router.message(Command("graph"), NTKChatFilter())
async def send_stats(message: types.Message, bot: Bot):
    """Send the occupancy graph: real data, predicted median and the p10–p90 range."""
    fig_visits, _ = await plotGraph.daily_graph_with_predictions()

    buffer_visits = io.BytesIO()
    fig_visits.savefig(buffer_visits, format="png")
    buffer_visits.seek(0)

    await bot.send_photo(
        chat_id=message.chat.id,
        photo=types.BufferedInputFile(
            file=buffer_visits.read(),
            filename="visits.png",
        ),
    )
    await message.delete()


@router.message(Command("learn"), SuperAdmins())
async def learn_models(msg: types.Message):
    """Learn models"""
    await predictModels.learn_models()
    await msg.answer("Models learned!")
    await msg.delete()


@router.message(Command("data"), SuperAdmins())
async def send_data(msg: types.Message, bot: Bot):
    """Send the occupancy series exported from SQLite."""
    input_file = types.BufferedInputFile(file=db.export_text(), filename="ntk_data.txt")
    await bot.send_document(msg.chat.id, input_file)
