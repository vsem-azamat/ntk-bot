"""One-off importer: legacy flat ``ntk_data.txt`` -> SQLite.

Usage (inside the container against the /data volume):
    python scripts/import_legacy_ntk_data.py /data/legacy.txt
"""

import sys
from datetime import datetime

from bot import db

_TS_FORMAT = "%Y-%m-%d %H:%M"


def import_file(path: str) -> tuple[int, int]:
    """Import every valid ``'<ts> - <count>'`` line. Returns (imported, skipped)."""
    db.init_db()
    imported = 0
    skipped = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                skipped += 1
                continue
            try:
                ts_part, count_part = row.split(" - ", 1)
                ts = datetime.strptime(ts_part.strip(), _TS_FORMAT)
                people = int(count_part.strip())
            except (ValueError, IndexError):
                skipped += 1
                continue
            db.insert_occupancy(ts, people)
            imported += 1
    return imported, skipped


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/import_legacy_ntk_data.py <ntk_data.txt>")
        raise SystemExit(2)
    imported, skipped = import_file(sys.argv[1])
    total = len(db.iter_rows())
    print(f"imported={imported} skipped={skipped} total_rows_in_db={total}")


if __name__ == "__main__":
    main()
