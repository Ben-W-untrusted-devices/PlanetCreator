"""Resumable HTTP downloads into ``data/raw/<dataset>/``."""

from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm

from .sources import Source

CHUNK = 1 << 20


def raw_path(source: Source, data_dir: Path) -> Path:
    return data_dir / "raw" / source.subdir / source.filename


def download(source: Source, data_dir: Path, force: bool = False) -> Path:
    """Download ``source`` if missing, resuming a partial ``.part`` file if present."""
    dest = raw_path(source, data_dir)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() and not force else 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(source.url, headers=headers, stream=True, timeout=60) as r:
        if have and r.status_code != 206:
            # Server ignored the range; start over.
            have = 0
            r.raise_for_status()
        elif not have:
            r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have or source.size_bytes
        mode = "ab" if have else "wb"
        with (
            open(part, mode) as f,
            tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=source.name) as bar,
        ):
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)
    return dest
