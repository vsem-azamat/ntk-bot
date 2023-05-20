import time
import asyncio
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from aiogram.utils.markdown import hlink

from collect_time import generaet_time_list


async def get_ntk_quantity():
    url = 'https://www.techlib.cz/en/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    body = soup.find_all('div', class_='panel-body text-center lead')
    return body[0].text.strip()

async def get_duplex_events() -> str:
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

async def recieve_ntk_data():
    delta_time = 20
    time_list = await generaet_time_list(delta_time)

    while True:
        current_time = datetime.now().strftime("%H:%M")
        if current_time in time_list:
            with open('ntk_data.txt', 'a') as file:
                date = datetime.now().strftime("%Y-%m-%d")
                quantity_ntk = await get_ntk_quantity()
                file.write(f"{date} {current_time} - {quantity_ntk}\n")
            await asyncio.sleep(delta_time*60-60)
        else:
            await asyncio.sleep(1)
