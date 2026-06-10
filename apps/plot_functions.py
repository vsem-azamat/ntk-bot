from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from apps.predictModels import predictModels

# Minimal "clean" palette — one ink tone, one accent, muted secondary text.
_INK = "#111827"  # real data line
_ACCENT = "#2563EB"  # prediction
_MUTED = "#9CA3AF"  # ticks / secondary text
_GRID = "#ECECEC"

_RU_WEEKDAYS = (
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
)
_RU_MONTHS = (
    "янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек",
)


def _ru_date(dt: datetime) -> str:
    return f"{_RU_WEEKDAYS[dt.weekday()]}, {dt.day} {_RU_MONTHS[dt.month - 1]}"


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

    async def daily_graph_with_predictions(
        self, target_day: datetime | None = None
    ) -> tuple[Figure, Axes]:
        """Minimal occupancy chart: real data + prediction median + p10–p90 band."""
        target_day = target_day or datetime.now()
        start = target_day.replace(
            hour=10 if target_day.isoweekday() >= 6 else 8,
            minute=0,
            second=0,
            microsecond=0,
        )
        end = start + timedelta(hours=18)

        x_times, y_quantities = await self.get_ntk_data(start, end)
        forecast = await predictModels.predict_day(target_day)

        fig, ax = plt.subplots(figsize=(10, 5.4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(True, axis="y", color=_GRID, linewidth=1.0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=_MUTED, labelsize=9)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # Forecast: soft p10–p90 band + median line.
        ax.fill_between(
            forecast.timestamps,
            forecast.p10,
            forecast.p90,
            color=_ACCENT,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            forecast.timestamps,
            forecast.p50,
            color=_ACCENT,
            linewidth=2.0,
            label="прогноз",
            zorder=2,
        )
        # Real data so far.
        ax.plot(
            x_times,
            y_quantities,
            color=_INK,
            linewidth=2.2,
            solid_capstyle="round",
            label="сейчас",
            zorder=3,
        )
        # Tag the latest real value at the tip of the line.
        if y_quantities:
            ax.annotate(
                f" {y_quantities[-1]}",
                (x_times[-1], y_quantities[-1]),
                color=_INK,
                fontsize=11,
                fontweight="bold",
                va="center",
                zorder=4,
            )

        ax.set_ylim(bottom=0)
        ax.set_xlim(start, end)
        ax.set_title(f"НТК · {_ru_date(start)}", loc="left", color=_INK, fontsize=13, pad=12)
        ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=_MUTED, handlelength=1.3)
        fig.tight_layout()
        return fig, ax


plotGraph = PlotGraphs()
