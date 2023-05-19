# FOR SCRAPING
import requests
from bs4 import BeautifulSoup

# TELEGRAM BOT API
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters import Command, BoundFilter
from aiogram.utils.markdown import hlink
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# OTHER
import os
from dotenv import load_dotenv

# GET BOT TOKEN
load_dotenv()
BOT_TOKEN = str(os.getenv('BOT_TOKEN'))

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)


# Filter

class NtkGroup(BoundFilter):
    async def check(self, message: types.Message) -> bool:
        return message.chat.id in [-1001684546093, -1001384533622]


def setup(dp: Dispatcher):
    dp.filters_factory.bind(NtkGroup)


# FUNCTIONS ----------------------

def get_ntk_quantity():
    url = 'https://www.techlib.cz/en/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    body = soup.find_all('div', class_='panel-body text-center lead')
    return body[0].text.strip()


def get_duplex_events() -> str:
    url = 'https://www.duplex.cz/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    d_list = soup.find_all('div', class_='col-sm-6 col-md-4 col-lg-3 archive-event')

    text = """💃<b>Duplex events:💃</b>"""
    for i in d_list:
        event_title = i.find('div', class_='event_title').text
        event_link = i.find('a', class_='event_title_link clearfix', href=True)['href']
        text += hlink(f'\n\n🎤{event_title}', event_link)
    return text


def gen_graf():
    pass


# BOT HANDLERS ---------------------
@dp.message_handler(Command("ntk", prefixes='!/'), NtkGroup())
async def ntk(msg: types.Message):
    q = get_ntk_quantity()
    text = f'📚В NTK сейчас людей: {q}'
    if int(q) >= 700:
        text += '\nДохуя крч.'
    await msg.answer(text)
    await msg.delete()


@dp.message_handler(Command('help', prefixes='!/'), NtkGroup())
async def ask_ntk(msg: types.Message):
    text = """<b>Хай, я создан для чата @chat_ntk!</b>

Команды:
/ntk - Показать кол-во людей в NTK
/duplex - Duplex Events <i>(без комментариев)</i>
/anon <text> -  Прислать в @chat_ntk анонимку
/int - Попросить анонимно инсту.

GitHub: github.com/vsem-azamat/ntk_bot
admin: t.me/vsem_azamat
    """
    await msg.answer(text, disable_web_page_preview=True)
    await msg.delete()



@dp.message_handler(Command('inst', prefixes='!/'))
async def get_me_inst(msg: types.Message):
    if msg.reply_to_message:
        text = "Дай инст, пожалуйста. \nМне для друга бота. \nОчень понравились :3"
        await msg.reply_to_message.reply(text)
    await msg.delete()


@dp.message_handler(Command('anon', prefixes='!/'))
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
        

@dp.message_handler(Command('анон', prefixes='!/'))
async def anon_message(msg: types.Message):
    black_list_of_users = [2132881105, ]
    text_anon = msg.text[6:]

    if msg.chat.id == msg.from_user.id and msg.from_user.id not in black_list_of_users:
        if 1 < len(text_anon) < 1000:
            # filter for bad words
            bad_words = [
                    "шлюх", "хохол", "рашист", 
                    "рашк", "шкур", "уеб", "блядь", 
                    "русн", "хохл", "шалав",
                    "целка", "целки"
                    ]
            for word in bad_words:
                if word in text_anon:
                    await bot.send_message(
                            chat_id=268388996,
                            text=f"id: {msg.from_user.id} \nlogin: @{msg.from_user.username} \n\ntext:\n{text_anon}"
                            )
                    await msg.answer("Ты что-то написал плохое((\nЯ вынужден рассказать это админу.")
                    return
            text = "💌<b>Анон плс:</b>\n\n" + text_anon
            await bot.send_message(chat_id=-1001684546093, text=text)
            await msg.reply('💌Отправлено!')
        else: await msg.answer('Текст слишком короткий, либо наоборот слишком большой!')
    else: await msg.delete()


@dp.message_handler(Command('duplex', prefixes='!/'), NtkGroup())
async def gen_duplex(msg: types.Message):
    await msg.answer(get_duplex_events(), disable_web_page_preview=True)
    await msg.delete()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
