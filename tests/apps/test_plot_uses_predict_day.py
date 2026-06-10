from datetime import datetime

from apps.predictModels import DayForecast


async def test_daily_graph_with_predictions_draws_predicted_median(monkeypatch):
    import apps.plot_functions as pf

    async def fake_get_ntk_data(self, start, end):
        return [start], [123]

    async def fake_predict_day(self, target_day=None):
        ts = [datetime(2024, 3, 1, 8, 0), datetime(2024, 3, 1, 9, 0)]
        return DayForecast(timestamps=ts, p10=[10, 20], p50=[15, 25], p90=[20, 30])

    monkeypatch.setattr(pf.PlotGraphs, "get_ntk_data", fake_get_ntk_data)
    monkeypatch.setattr(pf.predictModels, "predict_day", fake_predict_day.__get__(pf.predictModels))

    fig, ax = await pf.plotGraph.daily_graph_with_predictions(datetime(2024, 3, 1, 12, 0))

    labels = [line.get_label() for line in ax.get_lines()]
    assert "прогноз" in labels
    assert "сейчас" in labels
    import matplotlib.pyplot as plt

    plt.close(fig)
