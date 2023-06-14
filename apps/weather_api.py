import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class WeatherAPI:
    def __init__(self) -> None:
        self.url = 'https://api.open-meteo.com/v1/forecast'
        self.default_params = {
            'latitude': 50.1038,
            'longitude': 14.3906,
            'timezone': 'Europe/Berlin'
        }
        self.markers_path = 'icons/'


    def set_custom_marker(self, ax: Axes, x: list, y: list, marker_name: str) -> None:
        """
        Set custom marker on the plot
        
        markers: ['sun', 'rain', 'snowfall', 'wind']
        """
        marker = OffsetImage(plt.imread(f'{self.markers_path}{marker_name}.png'), zoom=0.08)
        for xi, yi in zip(x, y):
            ab = AnnotationBbox(marker, (xi, yi), frameon=False)
            ax.add_artist(ab)
    
    
    async def __get_weather_data(self, params: Optional[dict] = None) -> dict:
        params = params or  {}
        response = requests.get(url=self.url, params=self.default_params | params)
        data = response.json()
        return data


    async def get_current_weather(self) -> dict:
        """Get current weather data"""
        params = {'current_weather': True}
        return await self.__get_weather_data(params=params)
    
    
    async def get_weather_forecast(self, days: Optional[int] = None) -> dict:
        """
        Get weather forecast data
        
        forecast_days: 1-14
        hoyrly:
            temperature_2m,
            precipitation_probability,
            rain,
            showers,
            snowfall,
            windspeed_10m
        """
        params = {
            'hourly': 'temperature_2m,precipitation_probability,rain,showers,snowfall,windspeed_10m',
            'forecast_days': days or 1
            }
        return await self.__get_weather_data(params=params)
    
    
    async def __slice_zero_values(self, x: list, y: list) -> Tuple[list, list]:
        zero_indices = np.where(np.array(y) != 0)[0]
        return np.array(x)[zero_indices], np.array(y)[zero_indices]
    
    async def __slice_time_interval(self, x: list, y: list, start_datetime: datetime, end_datetime: datetime) -> Tuple[list, list]:
        indices = np.where(np.logical_and(start_datetime <= np.array(x), np.array(x) <= end_datetime))[0]
        return np.array(x)[indices], np.array(y)[indices]
    

    async def plot_daily_weather_forecast(self) -> Tuple[Figure, Axes, Axes]:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.05, left=0.1, right=0.9, top=0.95, bottom=0.1)
        ax1_2 = ax1.twinx()
        ax2_2 = ax2.twinx()
        
        for ax1.ax in [ax1, ax1_2]:
            ax1.ax.label_outer()
        
        data = await self.get_weather_forecast(1)
        hourly = data['hourly']
        datetimes = [datetime.strptime(time, '%Y-%m-%dT%H:%M') for time in hourly['time']]
        start_datetime = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)-timedelta(minutes=30)
        end_datetime = start_datetime + timedelta(hours=16)
        
        # Setting the 1 axes
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linewidth=0.3, which='both', axis='both')
        ax1.set_xlim([start_datetime, end_datetime]) # type: ignore
        ax1.set_ylabel('Temperature [ °C ]')   
        
        ax1_2.set_ylim(0, 20)
        
        ax2.set_xlim([start_datetime, end_datetime]) # type: ignore
        ax1_2.set_ylabel('Windspeed [ m/s ]')
        
        # Setting the 2 axes
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linewidth=0.3, which='both', axis='both')
        ax2.set_ylabel('Precipitation [ mm ]')
        ax2.set_xlabel('Time')
        
        ax2_2.set_ylabel('Precipitation probability [ % ]')
        ax2_2.set_ylim(0, 110)
        
        
        # Slice the arrays to the given datetime interval                
        x_datetimes, y_temperature_2m = await self.__slice_time_interval(datetimes, hourly['temperature_2m'], start_datetime, end_datetime)
        _, y_windspeed_10m = await self.__slice_time_interval(datetimes, hourly['windspeed_10m'], start_datetime, end_datetime)
        _, y_precipitation_probability = await self.__slice_time_interval(datetimes, hourly['precipitation_probability'], start_datetime, end_datetime)
        
        _, y_rain = await self.__slice_time_interval(datetimes, hourly['rain'], start_datetime, end_datetime)
        _, y_showers = await self.__slice_time_interval(datetimes, hourly['showers'], start_datetime, end_datetime)
        _, y_snowfall = await self.__slice_time_interval(datetimes, hourly['snowfall'], start_datetime, end_datetime)
        
        # Slice arrays from zero values
        x_datetimes_rain, y_rain = await self.__slice_zero_values(x_datetimes, y_rain)
        x_datetimes_showers, y_showers = await self.__slice_zero_values(x_datetimes, y_showers)
        x_datetimes_snowfall, y_snowfall = await self.__slice_zero_values(x_datetimes, y_snowfall)

        # PLOT THE DATA
        # First diagram
        ax1.bar(x_datetimes, y_temperature_2m, color='gold', width=0.03, label='Temperature [ °C ]')
        for x, y in zip(x_datetimes, y_temperature_2m):
            ax1.text(x, y, y, ha='center', va='bottom', fontsize=8)        
        ax1_2.plot(x_datetimes, y_windspeed_10m, linestyle='-', color='aqua', linewidth='1', label='Windspeed [ m/s ]')

        self.set_custom_marker(ax1, x_datetimes, [y*0.98 for y in y_temperature_2m], 'sun')
        self.set_custom_marker(ax1_2, x_datetimes, y_windspeed_10m, 'wind')
            
        # Second diagram
        if x_datetimes_rain:
            ax2.stackplot(x_datetimes, y_precipitation_probability, color='cyan', alpha=0.5, labels=['Rain [ mm ]'])
            self.set_custom_marker(ax2, x_datetimes_rain, y_rain, 'rain')
        
        if x_datetimes_showers:
            ax2.stackplot(x_datetimes_showers, y_showers, color='deepskyblue', alpha=0.5, labels=['Showers [ mm ]'])
        
        if x_datetimes_snowfall:
            ax2.stackplot(x_datetimes_snowfall, y_snowfall, color='white', alpha=0.5, labels=['Snowfall [ mm ]'])
            self.set_custom_marker(ax2_2, x_datetimes_snowfall, y_snowfall, 'snowfall')

        ax2_2.plot(x_datetimes, y_precipitation_probability, marker='+', linestyle='-', color='gray', linewidth='1', markersize='5', label='Precipitation probability [ % ]')
    
        
        # Set the title and legend
        ax1.set_title(f"Weather for NTK: {start_datetime.strftime('%A')} {start_datetime.strftime('%d-%m-%Y')}")
        ax1.legend(loc='upper left')
        ax1.set_ylim(min(y_temperature_2m), max(y_temperature_2m)+2)
        ax1_2.legend(loc='upper right')
        ax1_2.axvline(x=datetime.now(), color='black', linestyle='--', linewidth='0.7', alpha=0.5)
        
        ax2.legend(loc='upper left')
        ax2_2.legend(loc='upper right')
        ax2_2.axvline(x=datetime.now(), color='black', linestyle='--', linewidth='0.7', alpha=0.5)
        
        return fig, ax1, ax2
    

weatherAPI = WeatherAPI()
