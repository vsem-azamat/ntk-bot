import joblib
import numpy as np
from datetime import datetime
from typing import List, Union, TypeAlias, Tuple, Any

from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


ModelsML: TypeAlias = Union[RandomForestRegressor, GradientBoostingRegressor]

class PredictModels:
    async def perform_regression(self, data: List[str], modelML: ModelsML) -> Tuple[ModelsML, Union[Any, float, np.ndarray]]:  
        data = await self.remove_zero_values(data)  
        datetime_objects = [datetime.strptime(row.split(' - ')[0], "%Y-%m-%d %H:%M") for row in data]

        X_day_of_year = [dt.timetuple().tm_yday for dt in datetime_objects]
        X_day_of_week = [dt.weekday() for dt in datetime_objects]
        X_total_minutes = [(dt.hour * 60 + dt.minute) for dt in datetime_objects]
        X_month = [dt.month for dt in datetime_objects]

        X = np.column_stack((X_day_of_year, X_day_of_week, X_total_minutes, X_month))
        Y = np.array([int(row.split(" - ")[1]) for row in data])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

        
        modelML.fit(X_train, Y_train)
        joblib.dump(modelML, f'model_{modelML}.pkl')

        y_pred = modelML.predict(X_test)
        mse = mean_squared_error(Y_test, y_pred)
        
        return modelML, mse


    async def remove_zero_values(self, data: List[str]) -> List[str]:
        values = [int(row.split(' - ')[1].strip()) for row in data]
        indexes = []
        for i, value in enumerate(values):
            if value == 0:
                indexes.append(i)
        for index in indexes[::-1]: 
            data.pop(index)
        return data


    async def learn_models(self) -> None:
        data = []
        with open('ntk_data.txt', 'r') as file:
            for row in file:
                data.append(row.strip())
        if len(data) > 10:
            rf_regressor = RandomForestRegressor()
            await self.perform_regression(data, rf_regressor)

            gb_regressor = GradientBoostingRegressor()
            await self.perform_regression(data, gb_regressor)


   
    async def predict(self, model: ModelsML, new_data: List[str]) -> List[int]:
        datetime_objects = [datetime.strptime(row.split(' - ')[0], "%Y-%m-%d %H:%M") for row in new_data]

        X_day_of_year = [dt.timetuple().tm_yday for dt in datetime_objects]
        X_day_of_week = [dt.weekday() for dt in datetime_objects]
        X_total_minutes = [(dt.hour * 60 + dt.minute) for dt in datetime_objects]
        X_month = [dt.month for dt in datetime_objects]

        X = np.column_stack((X_day_of_year, X_day_of_week, X_total_minutes, X_month))

        y_pred = model.predict(X)

        return list(map(int, y_pred))


predictModels = PredictModels()
