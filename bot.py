# FOR SCRAPING
import requests
from bs4 import BeautifulSoup

# TELEGRAM BOT API
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters import Command, BoundFilter

# OTHER
import os
from dotenv import load_dotenv
import csv

# GET BOT TOKEN
load_dotenv()
BOT_TOKEN = str(os.getenv('BOT_TOKEN'))

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)
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


# BOT HANDLERS ---------------------
@dp.message_handler(Command("ntk", prefixes='!/'), NtkGroup())
async def ntk(msg: types.Message):
    q = get_ntk_quantity()
    text = f'В NTK сейчас людей: {q}'
    if int(q) >= 700:
        text = text + '\nДохуя крч.'
    await msg.answer(text)
    await msg.delete()


@dp.message_handler(Command('help', prefixes='!/'), NtkGroup())
async def ask_ntk(msg: types.Message):
    text = """
    Хай, моя задача в этой жизни показывать количество людей в НТК!
Используй комманду '/ntk'

GitHub: github.com/vsem-azamat/ntk_bot
admin: t.me/vsem_azamat
    """
    await msg.answer(text, disable_web_page_preview=True)
    await msg.delete()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
