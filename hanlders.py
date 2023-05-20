from datetime import datetime, timedelta

from aiogram import types
from aiogram.dispatcher.filters import Command

from bot import dp, bot
from configs import config, NtkGroup, BlackList 
from parse_functions import get_duplex_events, get_ntk_quantity, make_day_graph


@dp.message_handler(Command("ntk", prefixes='!/'), NtkGroup())
async def ntk(msg: types.Message):
    q = await get_ntk_quantity()
    text = f'📚В NTK сейчас людей: {q}'
    if int(q) >= 700:
        text += '\nДохуя крч.'
    await msg.answer(text)
    await msg.delete()


@dp.message_handler(Command('help', prefixes='!/'), NtkGroup())
async def ask_ntk(msg: types.Message):
    text = \
    "<b>Хай, я создан для чата @chat_ntk!</b>\n\n"\
    "Команды:\n"\
    "/ntk - Показать кол-во людей в NTK\n"\
    "/duplex - Duplex Events <i>(без комментариев)</i>\n"\
    "/anon <text> -  Прислать в @chat_ntk анонимку\n"\
    "/int - Попросить анонимно инсту.\n\n"\
    "admin: t.me/vsem_azamat"

    await msg.answer(text, disable_web_page_preview=True)
    await msg.delete()


@dp.message_handler(Command('inst', prefixes='!/'), BlackList())
async def get_me_inst(msg: types.Message):
    if msg.reply_to_message:
        text = "Дай инст, пожалуйста. \nМне для друга бота. \nОчень понравились :3"
        await msg.reply_to_message.reply(text)
    await msg.delete()


@dp.message_handler(Command('anon', prefixes='!/'), BlackList())
async def anon_old(msg: types.Message):
    text = \
        "<b>Привет! Я специально поменял команду, чтобы предупредить тебя о вынужденых нововведениях...</b>\n\n"\
        "Теперь абсолютная анонимность отсутствует! В боте есть потенциальный список black слов, "\
        "при обнаружении которых, бот перехватывает сообщение и шлёт его мне, а не в чат - "\
        "и только в этом случае я смогу увидеть отправителя.\n\n"\
        "Если в вашем сообщение отсутствуют маты (оскорбления), можете не беспокоиться!\n\n"\
        "<b>Новая команда:</b>\n"\
        "/анон 'тут анонимный текст'"

    await msg.answer(text)
        

@dp.message_handler(Command('анон', prefixes='!/'), BlackList())
async def anon_message(msg: types.Message):    
    text_anon = msg.text[6:]

    if msg.chat.id == msg.from_user.id:
        if 1 < len(text_anon) < 1000:
            # filter for bad words
            for word in config.BAD_WORDS:
                if word in text_anon:
                    await bot.send_message(
                            chat_id=268388996,
                            text=f"id: {msg.from_user.id} \nlogin: @{msg.from_user.username} \n\ntext:\n{text_anon}"
                            )
                    await msg.answer("Ты что-то написал плохое((\nЯ вынужден рассказать это админу.")
                    return
            text = "💌<b>Анон плс:</b>\n\n" + text_anon
            await bot.send_message(chat_id=config.ID_NTK_BIG_CHAT, text=text)
            await msg.reply('💌Отправлено!')
        else: await msg.answer('Текст слишком короткий, либо наоборот слишком большой!')
    else: await msg.delete()


@dp.message_handler(Command('duplex', prefixes='!/'), NtkGroup())
async def gen_duplex(msg: types.Message):
    text = await get_duplex_events()
    await msg.answer(text, disable_web_page_preview=True)
    await msg.delete()


@dp.message_handler(Command('graph', prefixes='!/'))
async def send_stats(msg: types.Message):
    image = await make_day_graph()
    await bot.send_photo(
        chat_id=msg.from_user.id,
        photo=types.InputFile(image, filename='graph')
    )
    