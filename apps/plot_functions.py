import io
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

from aiogram import Bot, types

import joblib
import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from config import config
from apps.collect_time import generaet_datetime_list
from apps.predictModels import predictModels


class PlotGraphs():

    async def get_ntk_data(self, start_datetime: datetime, end_datetime: datetime) -> Tuple[List[datetime], List[int]]:
        """Get data from file"""
        with open('ntk_data.txt', 'r') as file:
            datetimes = []
            quantities = []
            data = await predictModels.remove_zero_values([row  for row in file])
            for row in data:
                row_datetime = datetime.strptime(row.split(' - ')[0], "%Y-%m-%d %H:%M")
                if start_datetime <= row_datetime <= end_datetime:
                    quantities.append(int(row.split(' - ')[1].strip()))
                    datetimes.append(row_datetime)
        return datetimes, quantities


    async def daily_graph(self, target_day: Optional[datetime] = None) -> Tuple[Figure, Axes, datetime, datetime]:
        hours_delta = 18
        target_day = target_day or datetime.now()
        start_datetime = target_day.replace(hour=10 if target_day.isoweekday() >= 6 else 8).replace(minute=0, second=0, microsecond=0)
        end_datetime = start_datetime + timedelta(hours=hours_delta)
        
        # get data
        x_times, y_quantities = await self.get_ntk_data(start_datetime, end_datetime)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_yticks(range(0,1100,100))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        ax.plot(x_times, y_quantities, marker='o', linestyle='-', color='black', linewidth='1', markersize='5', label='Real data')               

        # anntation settings        
        ax.legend(loc='upper right')
        ax.set_xlabel('time')
        ax.set_ylabel('people')
        ax.set_title(f"NTK: {start_datetime.strftime('%A')} {start_datetime.strftime('%d-%m-%Y')}")
        ax.grid(True, linewidth=0.3, which='both', axis='both')

        
    
        ax.set_xlim([start_datetime-timedelta(minutes=30), end_datetime+timedelta(minutes=30)]) # type: ignore
        plt.xticks(rotation=45)

        return fig, ax, start_datetime, end_datetime


    async def add_daily_prediction(self, fig: Figure, ax: Axes, start_datetime: datetime, end_datetime: datetime, model_name: Optional[str] = None) -> Tuple[Figure, Axes, datetime, datetime]:
        datetime_objects =  await generaet_datetime_list(start_datetime, end_datetime, delta_minutes=10)
        x_day_of_week = [dt.weekday() for dt in datetime_objects]
        x_total_minutes = [(dt.hour * 60 + dt.minute) for dt in datetime_objects]
        x_month = [dt.month for dt in datetime_objects]

        match model_name:
            case 'GradientBoostingRegressor':
                model = joblib.load('model_GradientBoostingRegressor().pkl')
                color = 'red'
            case 'RandomForestRegressor':
                model = joblib.load('model_RandomForestRegressor().pkl')
                color = 'blue'
            case 'LinearRegression':
                model = joblib.load('model_LinearRegression().pkl')
                color = 'green'
            case _:
                model = joblib.load('model_GradientBoostingRegressor().pkl')
                color = 'red'

        x = np.column_stack((x_day_of_week, x_total_minutes, x_month))
        y = list(map(int, model.predict(x)))
        ax = fig.axes[0]

        ax.plot(datetime_objects, y, marker='+', linestyle='-', color=color, linewidth='1', markersize='3', label=str(model)[:-2], alpha=0.5)
        ax.legend()

        return fig, ax, start_datetime, end_datetime
    

    async def daily_graph_with_predictions(self, target_day: Optional[datetime] = None) -> Tuple[Figure, Axes]:
        target_day = target_day or datetime.now()
        fig, ax, start_datetime, end_datetime = await self.daily_graph(target_day)
        await self.add_daily_prediction(fig=fig, ax=ax, start_datetime=start_datetime, end_datetime=end_datetime, model_name='GradientBoostingRegressor')
        await self.add_daily_prediction(fig=fig, ax=ax, start_datetime=start_datetime, end_datetime=end_datetime, model_name='RandomForestRegressor')
        return fig, ax


plotGraph = PlotGraphs()
