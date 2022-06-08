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
    text = """<b>Хай, я создан для чата @chat_nkt!</b>

Команды:
/ntk - Показать кол-во людей в NTK
/duplex - Duplex Events <i>(без комментариев)</i>,.

GitHub: github.com/vsem-azamat/ntk_bot
admin: t.me/vsem_azamat
    """
    await msg.answer(text, disable_web_page_preview=True)
    await msg.delete()


@dp.message_handler(Command('duplex', prefixes='!/'), NtkGroup())
async def gen_duplex(msg: types.Message):
    await msg.answer(get_duplex_events(), disable_web_page_preview=True)
    await msg.delete()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
