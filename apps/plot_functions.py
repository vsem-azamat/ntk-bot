from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from apps.predictModels import predictModels


class PlotGraphs:
    async def get_ntk_data(
        self, start_datetime: datetime, end_datetime: datetime
    ) -> tuple[list[datetime], list[int]]:
        """Get occupancy data from SQLite within the given datetime range."""
        from bot import db

        datetimes: list[datetime] = []
        quantities: list[int] = []
        for row_datetime, count in db.fetch_occupancy():
            if count != 0 and start_datetime <= row_datetime <= end_datetime:
                datetimes.append(row_datetime)
                quantities.append(count)
        return datetimes, quantities

    async def daily_graph(
        self, target_day: datetime | None = None
    ) -> tuple[Figure, Axes, datetime, datetime]:
        hours_delta = 18
        target_day = target_day or datetime.now()
        start_datetime = target_day.replace(hour=10 if target_day.isoweekday() >= 6 else 8).replace(
            minute=0, second=0, microsecond=0
        )
        end_datetime = start_datetime + timedelta(hours=hours_delta)

        # get data
        x_times, y_quantities = await self.get_ntk_data(start_datetime, end_datetime)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_yticks(range(0, 1100, 100))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        ax.plot(
            x_times, y_quantities, linestyle="-", color="black", linewidth=2.5, label="Real data"
        )

        # annotation settings
        ax.legend(loc="upper right")
        ax.set_xlabel("time")
        ax.set_ylabel("people")
        ax.set_title(f"NTK: {start_datetime.strftime('%A')} {start_datetime.strftime('%d-%m-%Y')}")
        ax.grid(True, linewidth=0.3, which="both", axis="both")

        # Add annotation for max value
        if y_quantities:
            y_max = max(y_quantities)
            x_max = x_times[y_quantities.index(y_max)]
            ax.annotate(
                f"Max: {y_max}",
                xy=(x_max, y_max),
                xytext=(x_max, y_max * 0.9),
                bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black"),
            )

        ax.set_xlim(start_datetime - timedelta(minutes=30), end_datetime + timedelta(minutes=30))
        plt.xticks(rotation=45)

        return fig, ax, start_datetime, end_datetime

    async def daily_graph_with_predictions(
        self, target_day: datetime | None = None
    ) -> tuple[Figure, Axes]:
        target_day = target_day or datetime.now()
        fig, ax, _, _ = await self.daily_graph(target_day)

        forecast = await predictModels.predict_day(target_day)
        ax.plot(
            forecast.timestamps,
            forecast.p50,
            linestyle="-",
            color="darkgreen",
            linewidth=1.5,
            label="Prediction",
            zorder=0,
            alpha=0.7,
        )
        ax.legend(loc="upper right")
        return fig, ax


plotGraph = PlotGraphs()
