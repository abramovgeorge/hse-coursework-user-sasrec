import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from src.datasets.session_aware import SessionAwareDataset
from src.utils.io_utils import ROOT_PATH


class YambdaDataset(SessionAwareDataset):
    """
    Yambda-50m dataset
    """

    def __init__(
        self,
        *args,
        inter_type="likes",
        yambda_size="50m",
    ):
        """
        Args:
            inter_type (str): interaction type in Yambda dataset
            yambda_size (str): Yambda dataset size. Could be equal to 50m, 500m or 5b
        """
        self._inter_type = inter_type
        self._yambda_size = yambda_size
        super().__init__(*args)

    def _load_data(self):
        """
        Returns a specified version of the Yambda dataset
        """
        path = (
            ROOT_PATH
            / "data"
            / "yambda"
            / f"{self._yambda_size}"
            / f"{self._inter_type}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df = pd.read_parquet(path)
        else:
            ds = load_dataset(
                "yandex/yambda",
                data_dir=f"flat/{self._yambda_size}",
                data_files=f"{self._inter_type}.parquet",
            )
            df = ds["train"].to_pandas()
            df.to_parquet(path, index=False)
        if "played_ratio_pct" in df.columns:
            # we leave listens that lasted at least half of the track
            df = df[df["played_ratio_pct"] >= 50]
        return df
