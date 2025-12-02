import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets import load_dataset


class YambdaDataset(Dataset):
    """
    Yambda-50m dataset

    Uses like interaction between users and songs
    """

    def __init__(
        self,
        name="train",
        inter_type="likes",
        yambda_size="50m",
        inactivity_thresh=1800,
        q=0.8,
        min_inter_user=None,
        min_inter_item=None,
        min_len=None,
        limit=None,
        max_len=100,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            name (str): dataset partition name
            inter_type (str): interaction type in Yambda dataset
            yambda_size (str): Yambda dataset size. Could be equal to 50m, 500m or 5b
            inactivity_thresh (int): length of the inactivity window in seconds,
                used for splitting the continuous dataset into sessions
            q (float): fraction of train data
            min_inter_user (int | None): minimal number of interaction for a user
            min_inter_item (int | None): minimal number of interaction for an item
            min_len (int | None): minimal length of a session
            limit (int | None): if not None, limit the total number of sessions
                in the dataset to 'limit' elements.
            max_len (int | None): if not None, limit the sessions by last 'max_len' items
            shuffle_index (bool): if True, shuffle the index.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        ds = load_dataset(
            "yandex/yambda",
            data_dir=f"flat/{yambda_size}",
            data_files=f"{inter_type}.parquet",
        )
        df = ds["train"].to_pandas()
        if "played_ratio_pct" in df.columns:
            # we leave listens that lasted at least half of the track
            df = df[df["played_ratio_pct"] >= 50]
        item_vcs = df["item_id"].value_counts()
        user_vcs = df["uid"].value_counts()
        if min_inter_item is not None:
            df = df[df["item_id"].isin(item_vcs[item_vcs > min_inter_item].index)]
        if min_inter_user is not None:
            df = df[df["uid"].isin(user_vcs[user_vcs > min_inter_user].index)]
        df["uid"] = pd.Categorical(df["uid"]).codes
        df["item_id"] = pd.Categorical(df["item_id"]).codes
        self.n_users = df["uid"].max() + 1
        self.n_items = df["item_id"].max() + 1
        df = self._create_session_ids(df, inactivity_thresh)
        session_sizes = df["session_id"].value_counts()
        if min_len is not None:
            df = df[df["session_id"].isin(session_sizes[session_sizes > min_len].index)]
        train, test = self._train_test_split(df, q)
        if name == "train":
            self._df = train.copy()
            users = dict(zip(self._df["session_id"], self._df["uid"]))
            index = [
                {"session_id": session_id, "user": users[session_id]}
                for session_id in self._df["session_id"].unique()
            ]
        else:
            # we perform leave-one-out split on the test sessions
            test_items = test[~test.duplicated(subset="session_id", keep="last")]
            test_items = dict(zip(test_items["session_id"], test_items["item_id"]))
            test = test[test.duplicated(subset="session_id", keep="last")]
            self._df = test.copy()
            users = dict(zip(self._df["session_id"], self._df["uid"]))
            index = [
                {
                    "session_id": session_id,
                    "user": users[session_id],
                    "item": test_items[session_id],
                }
                for session_id in self._df["session_id"].unique()
            ]
        self._item_seqs = (
            self._df.groupby("session_id")["item_id"]
            .apply(lambda seq: torch.tensor(seq.values, dtype=torch.long)[-max_len:])
            .to_dict()
        )
        self._index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self.instance_transforms = instance_transforms
        self.item_counts = dict(self._df["item_id"].value_counts())

    @staticmethod
    def _create_session_ids(df, inactivity_thresh=1800):
        """
        Create session ids by inactivity threshold.

        Args:
            df (pd.DataFrame): dataset as a pandas dataframe.
            inactivity_thresh (int): threshold of inactivity for creating sessions.
        Returns:
            pd.DataFrame: dataset with added global session ids
        """
        df = df.copy()
        df["time_gap"] = df.groupby("uid")["timestamp"].diff()
        df["is_new_session"] = df["time_gap"] > inactivity_thresh
        df["user_session_id"] = df.groupby("uid")["is_new_session"].cumsum()
        df["session_id"] = df.groupby(["uid", "user_session_id"]).ngroup()
        return df.drop(columns=["time_gap", "is_new_session", "user_session_id"])

    @staticmethod
    def _train_test_split(df, q):
        """
        Split the dataset into train and test subsets via time split.

        Args:
            df (pd.DataFrame): dataset as a pandas dataframe, must contain session ids.
            q (float): fraction of train data.
        Returns:
            tuple(pd.DataFrame, pd.DataFrame): train and test subsets.
        """
        timestamp_q = np.quantile(df["timestamp"], q=q)
        session_times = df.groupby("session_id")["timestamp"].agg(["min", "max"])
        bad_sessions = session_times[
            (session_times["min"] < timestamp_q) & (session_times["max"] >= timestamp_q)
        ].index
        df = df[~df["session_id"].isin(bad_sessions)]
        train, test = (
            df[df["timestamp"] <= timestamp_q],
            df[df["timestamp"] > timestamp_q],
        )
        return train, test

    def __len__(self):
        """
        Get length of the dataset.

        Returns:
            int: number of unique sessions in the dataset
        """
        return len(self._index)

    def __getitem__(self, ind):
        """
        Get element from the dataset (i.e., session), preprocess it, and combine it into a dict.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        session_id = self._index[ind]["session_id"]
        data = self._item_seqs[session_id]
        item = self._index[ind].get("item", None)
        user = self._index[ind]["user"]
        instance_data = {"seq": data, "user": user}
        if item is not None:
            instance_data["item"] = item
        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
