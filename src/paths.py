"""Project and dataset path helpers."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/liuyanan/data/Research_Data/4D-UMS")


def get_data_root() -> Path:
    override = os.environ.get("UMS_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_DATA_ROOT


DATA_ROOT = get_data_root()


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)