# src/data_loader.py

import logging
from typing import Optional

import pandas as pd

from src.config import RAW_DATA_FILE

logger = logging.getLogger(__name__)


def load_raw_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw Spotify dataset from disk.

    Parameters
    ----------
    path : str, optional
        Path to raw data file. Defaults to RAW_DATA_FILE from config.

    Returns
    -------
    pd.DataFrame
        Loaded raw dataset.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If the dataset is empty or missing required columns.
    """
    from pathlib import Path

    data_path = Path(path) if path is not None else RAW_DATA_FILE


    logger.info(f"Loading raw data from {data_path}")

    # ---------- File existence check ----------
    if not data_path.exists():
        logger.error(f"Raw data file not found at {data_path}")
        raise FileNotFoundError(f"Raw data file not found at {data_path}")

    # ---------- Load data ----------
    df = pd.read_csv(data_path)

    # ---------- Empty dataset check ----------
    if df.empty:
        logger.error("Loaded dataset is empty")
        raise ValueError("Loaded dataset is empty")

    # ---------- Schema sanity check ----------
    expected_columns = {
        "track_id",
        "genre",
        "popularity"
    }

    missing_cols = expected_columns - set(df.columns)
    if missing_cols:
        logger.error(f"Missing expected columns: {missing_cols}")
        raise ValueError(f"Missing expected columns: {missing_cols}")

    logger.info(
        f"Raw data loaded successfully | shape={df.shape} | columns={len(df.columns)}"
    )

    return df
