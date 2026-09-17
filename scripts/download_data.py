#!/usr/bin/env python
"""Download the raw datasets. Usage: download_data.py [--months 200401 200407 ...]"""

import argparse
from pathlib import Path

from planetcreator.download import download
from planetcreator.sources import DEFAULT_MONTHS, default_sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--months", nargs="+", default=list(DEFAULT_MONTHS), help="BMNG months, YYYYMM")
    ap.add_argument("--list", action="store_true", help="print sources and exit")
    args = ap.parse_args()

    sources = default_sources(tuple(args.months))
    if args.list:
        for s in sources:
            mb = f"{s.size_bytes / 1e6:.0f} MB" if s.size_bytes else "? MB"
            print(f"{s.name:22s} {mb:>8s}  {s.license:22s} {s.url}")
        return
    for s in sources:
        print(download(s, args.data_dir))


if __name__ == "__main__":
    main()
