import random

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

from src.datasets.session_aware import SessionAwareDataset
from src.utils.io_utils import ROOT_PATH


class GowallaDataset(SessionAwareDataset):
    """
    Gowalla dataset
    https://snap.stanford.edu/data/loc-gowalla.html
    """

    def _load_data(self):
        """
        Returns the Gowalla dataset
        """
        path = ROOT_PATH / "data" / "gowalla.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df = pd.read_parquet(path)
        else:
            raw_path = ROOT_PATH / "data" / "loc-gowalla_totalCheckins.txt.gz"
            self._download_file(
                url="https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
                dest_path=raw_path,
            )
            df = pd.read_csv(
                raw_path,
                sep="\t",
                header=None,
                names=["uid", "timestamp", "latitude", "longitude", "item_id"],
                parse_dates=["timestamp"],
                compression="gzip",
            ).drop(columns=["latitude", "longitude"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["timestamp"] = df["timestamp"].astype("int64") // 10**9
            df = df.sort_values(by=["uid", "timestamp"]).reset_index(drop=True)
            df.to_parquet(path, index=False)
            raw_path.unlink(missing_ok=True)
        return df

    def _download_file(self, url, dest_path):
        """
        Download file from url into dest_path

        Args:
            url (str): URL of the file
            dest_path (Path): Destination
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
