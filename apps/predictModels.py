import asyncio
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


def parse_row_datetime(row: str) -> datetime:
    """Parse the timestamp from a '<YYYY-mm-dd HH:MM> - <count>' data row."""
    return datetime.strptime(row.split(" - ")[0], "%Y-%m-%d %H:%M")


def extract_features(datetimes: list[datetime]) -> np.ndarray:
    """Build the model feature matrix: day of year, weekday, minutes, month."""
    return np.column_stack(
        (
            [dt.timetuple().tm_yday for dt in datetimes],
            [dt.weekday() for dt in datetimes],
            [dt.hour * 60 + dt.minute for dt in datetimes],
            [dt.month for dt in datetimes],
        )
    )


class PredictModels:
    async def perform_regression(
        self, data: list[str], modelML: ModelsML
    ) -> tuple[ModelsML, float]:
        data = await self.remove_zero_values(data)
        X = extract_features([parse_row_datetime(row) for row in data])
        Y = np.array([int(row.split(" - ")[1]) for row in data])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

        # sklearn fit is CPU-bound; run it off the event loop so polling isn't blocked.
        await asyncio.to_thread(modelML.fit, X_train, Y_train)
        joblib.dump(modelML, model_filename(type(modelML).__name__))

        y_pred = modelML.predict(X_test)
        mse = mean_squared_error(Y_test, y_pred)

        return modelML, float(mse)

    async def remove_zero_values(self, data: list[str]) -> list[str]:
        rows = []
        for row in data:
            try:
                count = int(row.split(" - ")[1].strip())
            except (IndexError, ValueError):
                continue  # skip blank or malformed lines
            if count != 0:
                rows.append(row)
        return rows

    async def learn_models(self) -> None:
        try:
            with open(cnfg.NTK_DATA_PATH, encoding="utf-8") as file:
                data = [row.strip() for row in file]
        except FileNotFoundError:
            return
        if len(data) > 10:
            await self.perform_regression(data, RandomForestRegressor())
            await self.perform_regression(data, GradientBoostingRegressor())


predictModels = PredictModels()
