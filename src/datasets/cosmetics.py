from pathlib import Path

import kagglehub
import pandas as pd

from src.datasets.session_aware import SessionAwareDataset
from src.utils.io_utils import ROOT_PATH


class CosmeticsDataset(SessionAwareDataset):
    """
    Cosmetics dataset
    https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop
    We use only one month of the data
    """

    def _load_data(self):
        """
        Returns the Cosmetics dataset
        """
        path = ROOT_PATH / "data" / "cosmetics.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df = pd.read_parquet(path)
        else:
            kaggle_path = kagglehub.dataset_download(
                "mkechinov/ecommerce-events-history-in-cosmetics-shop"
            )
            kaggle_path = Path(kaggle_path)
            df = pd.read_csv(kaggle_path / "2019-Dec.csv")
            df = df[df.event_type == "view"]
            df = df.rename(
                columns={
                    "user_id": "uid",
                    "product_id": "item_id",
                    "event_time": "timestamp",
                }
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
            df = df[["timestamp", "uid", "item_id"]]
            df = df.sort_values(by=["uid", "timestamp"]).reset_index(drop=True)
            df.to_parquet(path, index=False)
        return df
