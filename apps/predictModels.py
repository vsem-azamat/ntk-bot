from datetime import datetime
from typing import TypeAlias

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from config import cnfg

ModelsML: TypeAlias = RandomForestRegressor | GradientBoostingRegressor


def model_filename(model_name: str) -> str:
    """Return the on-disk filename for a model given its class name."""
    return f"model_{model_name}.pkl"


class PredictModels:
    async def perform_regression(
        self, data: list[str], modelML: ModelsML
    ) -> tuple[ModelsML, float]:
        data = await self.remove_zero_values(data)
        datetime_objects = [
            datetime.strptime(row.split(" - ")[0], "%Y-%m-%d %H:%M") for row in data
        ]

        X_day_of_year = [dt.timetuple().tm_yday for dt in datetime_objects]
        X_day_of_week = [dt.weekday() for dt in datetime_objects]
        X_total_minutes = [(dt.hour * 60 + dt.minute) for dt in datetime_objects]
        X_month = [dt.month for dt in datetime_objects]

        X = np.column_stack((X_day_of_year, X_day_of_week, X_total_minutes, X_month))
        Y = np.array([int(row.split(" - ")[1]) for row in data])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

        modelML.fit(X_train, Y_train)
        joblib.dump(modelML, model_filename(type(modelML).__name__))

        y_pred = modelML.predict(X_test)
        mse = mean_squared_error(Y_test, y_pred)

        return modelML, float(mse)

    async def remove_zero_values(self, data: list[str]) -> list[str]:
        return [row for row in data if int(row.split(" - ")[1].strip()) != 0]

    async def learn_models(self) -> None:
        with open(cnfg.NTK_DATA_PATH, encoding="utf-8") as file:
            data = [row.strip() for row in file]
        if len(data) > 10:
            await self.perform_regression(data, RandomForestRegressor())
            await self.perform_regression(data, GradientBoostingRegressor())


predictModels = PredictModels()
