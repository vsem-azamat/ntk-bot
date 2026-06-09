from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import splev, splrep

from apps.collect_time import generate_datetime_list
from apps.predictModels import (
    extract_features,
    model_filename,
    parse_row_datetime,
    predictModels,
)
from config import cnfg

# Colour used to draw each model's prediction line.
MODEL_COLORS = {
    "GradientBoostingRegressor": "red",
    "RandomForestRegressor": "darkgreen",
}
DEFAULT_MODEL = "GradientBoostingRegressor"


class PlotGraphs:
    async def get_ntk_data(
        self, start_datetime: datetime, end_datetime: datetime
    ) -> tuple[list[datetime], list[int]]:
        """Get occupancy data from file within the given datetime range."""
        datetimes: list[datetime] = []
        quantities: list[int] = []
        try:
            with open(cnfg.NTK_DATA_PATH, encoding="utf-8") as file:
                data = await predictModels.remove_zero_values(list(file))
        except FileNotFoundError:
            return datetimes, quantities
        for row in data:
            row_datetime = parse_row_datetime(row)
            if start_datetime <= row_datetime <= end_datetime:
                quantities.append(int(row.split(" - ")[1].strip()))
                datetimes.append(row_datetime)
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

    async def add_daily_prediction(
        self,
        fig: Figure,
        ax: Axes,
        start_datetime: datetime,
        end_datetime: datetime,
        model_name: str | None = None,
    ) -> tuple[Figure, Axes, datetime, datetime]:
        x_datetime = await generate_datetime_list(start_datetime, end_datetime, delta_minutes=10)

        if model_name not in MODEL_COLORS:
            model_name = DEFAULT_MODEL
        color = MODEL_COLORS[model_name]

        # The model may not have been trained yet (e.g. too little data); skip its line.
        model_path = model_filename(model_name)
        if not Path(model_path).exists():
            return fig, ax, start_datetime, end_datetime
        model = joblib.load(model_path)

        xPredict = extract_features(x_datetime)
        y = list(map(int, model.predict(xPredict)))
        ax = fig.axes[0]

        # Interpolate data for smooth line
        x_numeric = np.array([dt.timestamp() for dt in x_datetime])
        x_interp_numeric = np.linspace(min(x_numeric), max(x_numeric), num=100)
        x_interp_datetime = [datetime.fromtimestamp(ts) for ts in x_interp_numeric]

        tck = splrep(x_numeric, y)
        y_interp = splev(x_interp_numeric, tck)

        ax.plot(
            x_interp_datetime,
            y_interp,
            linestyle="-",
            color=color,
            linewidth=1,
            label=model_name,
            zorder=0,
            alpha=0.5,
        )
        ax.legend()

        return fig, ax, start_datetime, end_datetime

    async def daily_graph_with_predictions(
        self, target_day: datetime | None = None
    ) -> tuple[Figure, Axes]:
        target_day = target_day or datetime.now()
        fig, ax, start_datetime, end_datetime = await self.daily_graph(target_day)
        for model_name in MODEL_COLORS:
            await self.add_daily_prediction(
                fig=fig,
                ax=ax,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                model_name=model_name,
            )
        return fig, ax


plotGraph = PlotGraphs()
