from urllib.parse import urlencode

import pandas as pd
import requests

from src.datasets.session_aware import SessionAwareDataset
from src.utils.io_utils import ROOT_PATH


class YelpDataset(SessionAwareDataset):
    """
    Yelp dataset
    https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
    Preprocessed version is from https://github.com/AIRI-Institute/Scalable-SASRec/
    """

    def _load_data(self):
        """
        Returns the Yelp dataset
        """
        path = ROOT_PATH / "data" / "yelp.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df = pd.read_parquet(path)
        else:
            public_key = "https://disk.yandex.ru/d/qdJZPjGt14H01w"
            base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
            final_url = base_url + urlencode(dict(public_key=public_key))
            response = requests.get(final_url)
            download_url = response.json()["href"]
            df = pd.read_csv(download_url).rename(
                columns={"userid": "uid", "itemid": "item_id"}
            )
            df = df.sort_values(by=["uid", "timestamp"]).reset_index(drop=True)
            df.to_parquet(path, index=False)
        return df
