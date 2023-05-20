from datetime import datetime, timedelta
from typing import List


async def generaet_time_list(delta_time: int = 20) -> List[str]:
    start_time = "08:00"
    end_time = "02:00"

    # Convert times to datetime objects
    start = datetime.strptime(start_time, "%H:%M")
    end = datetime.strptime(end_time, "%H:%M")

    # If end time is earlier than start time, assume it's next day
    if end < start:
        end += timedelta(days=1)

    # Initialize time list
    time_list = []

    # Add times to the list every delta_time minutes
    while start <= end:
        time_list.append(start.time().strftime("%H:%M"))
        start += timedelta(minutes=delta_time)

    return time_list

