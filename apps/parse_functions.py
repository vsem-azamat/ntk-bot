import aiohttp
from bs4 import BeautifulSoup

NTK_URL = "https://www.techlib.cz/en/"
# The site has gone silent for days at a time; fail fast instead of aiohttp's
# 5-minute default so /ntk answers and the collector logs the failure promptly.
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)


async def get_ntk_quantity() -> int:
    """Scrape the current number of people in the NTK from the library website."""
    async with (
        aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session,
        session.get(NTK_URL) as response,
    ):
        response.raise_for_status()
        soup = BeautifulSoup(await response.text(), "lxml")
        body = soup.find_all("div", class_="panel-body text-center lead")
        return int(body[0].text.strip())
