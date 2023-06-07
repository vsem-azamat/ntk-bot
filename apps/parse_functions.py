import requests
from datetime import datetime
from typing import Tuple, Generator

from bs4 import BeautifulSoup


async def get_ntk_quantity() -> int:
    url = 'https://www.techlib.cz/en/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    body = soup.find_all('div', class_='panel-body text-center lead')
    return int(body[0].text.strip())


async def get_values_ntk_visits(start_datetime: datetime, end_datetime: datetime):
    def read_file() -> Generator[Tuple[datetime, int, str], None, None]:
        with open('ntk_data.txt', 'r') as file:
            for row in file:
                day, time, _, count = row.split(' ')
                row_datetime = datetime.strptime(f'{day} {time}', '%Y-%m-%d %H:%M')
                if start_datetime <= row_datetime <= end_datetime:
                    yield row_datetime, int(count), row
    try:
        x_dates, y_values, XY = zip(*read_file())
        return x_dates, y_values, list(XY)
    except ValueError:
        return [], [], []
