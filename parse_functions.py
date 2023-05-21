import io
import asyncio
import requests
from datetime import datetime, timedelta
from typing import List

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from aiogram.utils.markdown import hlink

from configs import config
from collect_time import generaet_time_list


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


async def get_ntk_quantity() -> str:
    url = 'https://www.techlib.cz/en/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    body = soup.find_all('div', class_='panel-body text-center lead')
    return body[0].text.strip()


async def recieve_ntk_data(delta_time: int = 20):
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


async def get_values_ntk_visits(start_datetime: datetime, end_datetime: datetime) -> List[List]:
    def read_file() -> List[List]:
        with open('ntk_data.txt', 'r') as file:
            for row in file:
                day, time, _, count = row.split(' ')
                row_datetime = datetime.strptime(f'{day} {time}', '%Y-%m-%d %H:%M')
                count = int(count[:-1])
                if start_datetime <= row_datetime <= end_datetime:
                    yield row_datetime, count
    x_dates, y_values = zip(*read_file())
    return list(x_dates), list(y_values)               


async def make_day_graph(target_datetime: datetime = datetime.now()) -> io.BytesIO:
        start_datetime = target_datetime.replace(hour=10 if target_datetime.isoweekday() >= 6 else 8)
        end_datetime = start_datetime + timedelta(hours=16)

        x_dates, y_values = await get_values_ntk_visits(start_datetime, end_datetime)
        x_dates = [f'{str(time.hour).zfill(2)}:{str(time.minute).zfill(2)}' for time in x_dates]

        plt.figure(figsize=(10, 6))
        plt.plot(x_dates, y_values, marker='o', linestyle='-', color='black')

        plt.xlabel('time')
        plt.ylabel('people')
        plt.title(f"NTK: {start_datetime.strftime('%A')} {start_datetime.strftime('%d-%m-%Y')}")
        plt.xticks(rotation=45)
        
        x_axis_dates = [
            f'{str(start_datetime.hour).zfill(2)}:{str(start_datetime.minute).zfill(2)}' +
            timedelta(minutes=config.DELTA_TIME_FOR_RECIEVE_NTK*i)
            for i in range((end_datetime-start_datetime)//timedelta(minutes=config.DELTA_TIME_FOR_RECIEVE_NTK))
        ]
        y_zero = range(len(x_dates))
        
        plt.scatter(x_dates, y_zero, s=0, color='none')
        plt.xticks(x_axis_dates[::2])
        plt.yticks(range(0,1100,100))
        plt.grid(True, linewidth=0.5, which='both', axis='both')

        y_max = max(y_values)
        x_max = y_values.index(y_max)
        plt.annotate(
            text=f'Max: {y_max}', 
            xy=(x_max, y_max), 
            xytext=(x_max-1,y_max-100), 
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round')
        )
            
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return buffer
