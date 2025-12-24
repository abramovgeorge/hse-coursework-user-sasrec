import argparse

import optuna
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cn", required=True, help="Common name/prefix (e.g. sasrec)")
    return p.parse_args()


def main():
    args = parse_args()
    cn = args.cn

    src = JournalStorage(JournalFileBackend(f"./{cn}_optuna.log"))
    dst = RDBStorage(f"sqlite:///{cn}.db")

    dst_names = set(optuna.study.get_all_study_names(storage=dst))
    for name in optuna.study.get_all_study_names(storage=src):
        if name in dst_names:
            optuna.delete_study(study_name=name, storage=dst)
        optuna.copy_study(
            from_study_name=name,
            from_storage=src,
            to_storage=dst,
            to_study_name=name,
        )


if __name__ == "__main__":
    main()
