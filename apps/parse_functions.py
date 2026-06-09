import aiohttp
from bs4 import BeautifulSoup


async def get_ntk_quantity() -> int:
    """Scrape the current number of people in the NTK from the library website."""
    url = "https://www.techlib.cz/en/"
    async with aiohttp.ClientSession() as session, session.get(url) as response:
        soup = BeautifulSoup(await response.text(), "lxml")
        body = soup.find_all("div", class_="panel-body text-center lead")
        return int(body[0].text.strip())
