from datetime import datetime, timedelta
from typing import List


async def generaet_datetime_list(start_datetime: datetime, end_datetime: datetime, delta_minutes: int = 20) -> List[datetime]:
    time_list = []
    while start_datetime <= end_datetime:
        time_list.append(start_datetime)
        start_datetime += timedelta(minutes=delta_minutes)
    return time_list


async def generaet_time_list(delta_minutes: int = 20) -> List[str]:
    start_time = "08:00"
    end_time = "02:00"

    # Convert times to datetime objects
    start = datetime.strptime(start_time, "%H:%M")
    end = datetime.strptime(end_time, "%H:%M")

    # If end time is earlier than start time, assume it's next day
    if end < start:
        end += timedelta(days=1)

    time_list = []
    while start <= end:
        time_list.append(start.time().strftime("%H:%M"))
        start += timedelta(minutes=delta_minutes)

    return time_list
