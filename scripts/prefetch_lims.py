"""Cache BIS LIMS lab lists for common standards into data/lims_labs.json.

The chat looks up labs live when a standard is not cached; caching the
standards used in demos keeps answers fast and working offline.

Usage:
  PYTHONPATH=src python scripts/prefetch_lims.py            # default list
  PYTHONPATH=src python scripts/prefetch_lims.py 4151 "2347" "9873:1"

Each argument is an IS base number, optionally ``base:part``.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bis_assistant import labs  # noqa: E402

# Compulsory products that come up in demos and common questions.
DEFAULT = [
    "4151", "2347", "7466", "17526", "17790", "17803", "14543", "13340", "1293",
    "694", "9968:1", "16102:1", "10322:5", "15885:2", "16046:2", "13252:1",
    "302:1", "9873:1", "15644", "2062", "1786", "269", "8112", "12269", "4984",
    "14151:1", "10500", "13428", "1417", "2112",
]


def main() -> None:
    wanted = sys.argv[1:] or DEFAULT
    data = labs._load_json(labs.CACHE_PATH) or {"standards": {}}
    data.setdefault("standards", {})
    for item in wanted:
        base, _, part = item.partition(":")
        rows = labs.lookup(base, part, live=True, timeout_s=60, use_cache=False)
        if rows is None:
            print(f"IS {item}: lookup failed")
            continue
        data["standards"][labs._cache_key(base, part)] = {
            "fetched": date.today().isoformat(), "source_url": labs.lims_url(base, part),
            "rows": rows}
        print(f"IS {item}: {len(rows)} labs")
        time.sleep(1.5)
    data["source"] = labs.LIMS_SEARCH
    labs.CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
