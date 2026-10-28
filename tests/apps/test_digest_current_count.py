from datetime import datetime
from unittest.mock import patch

from apps.digest import current_count


def test_pre_opening_residue_is_not_reported_as_current():
    rows = [(datetime(2026, 10, 27, 1, 40), 205), (datetime(2026, 10, 27, 2, 0), 203)]
    with patch("bot.db.fetch_occupancy", return_value=rows):
        assert current_count(datetime(2026, 10, 27, 6, 45)) is None


def test_open_sample_is_reported():
    rows = [(datetime(2026, 10, 27, 9, 20), 243)]
    with patch("bot.db.fetch_occupancy", return_value=rows):
        assert current_count(datetime(2026, 10, 27, 9, 30)) == 243
