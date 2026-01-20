import argparse
import warnings

import optuna
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def run_train(config, trial, optuna_metric):
    """
    Pure callable training entrypoint: takes a composed Hydra config and returns final metrics.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "mps" if torch.backends.mps.is_available() else device
    else:
        device = config.trainer.device

    dataloaders, batch_transforms, data_kwargs = get_dataloaders(config, device)

    model = instantiate(
        config.model, loss_class=config.loss_function._target_, **data_kwargs
    ).to(device)
    logger.info(model)

    loss_function = instantiate(config.loss_function, **data_kwargs).to(device)

    metrics = {
        k: [instantiate(metric, **data_kwargs) for metric in v]
        for k, v in config.metrics.items()
    }

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        trial=trial,
        optuna_metric=optuna_metric,
    )
    return trainer.train()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--cn", default="baseline")
    p.add_argument("--metric", default="test_hitrate@10")
    return p.parse_args()


def main():
    args = parse_args()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(version_base=None, config_path="src/configs")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5,
    )

    def objective(trial):
        batch_size = trial.suggest_categorical(
            "batch_size",
            [32, 64, 128],
        )
        hidden_dim = trial.suggest_categorical(
            "hidden_dim",
            [32, 64, 128],
        )
        num_blocks = trial.suggest_int("num_blocks", 1, 6)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)
        n_buckets = trial.suggest_int("n_buckets", 32, 2048, log=True)
        bucket_size_y = trial.suggest_int("bucket_size_y", 32, 2048, log=True)
        mix_x = trial.suggest_categorical("mix_x", [True, False])

        cfg = compose(
            config_name=args.cn,
            overrides=[
                f"dataloader.batch_size={batch_size}",
                f"model.hidden_dim={hidden_dim}",
                f"model.num_blocks={num_blocks}",
                f"model.num_heads={num_heads}",
                f"model.dropout_rate={dropout_rate}",
                f"loss_function.n_buckets={n_buckets}",
                f"loss_function.bucket_size_x={n_buckets}",
                f"loss_function.bucket_size_y={bucket_size_y}",
                f"loss_function.mix_x={str(mix_x).lower()}",
            ],
        )

        try:
            results = run_train(cfg, trial=trial, optuna_metric=args.metric)
            return float(results[args.metric])
        except optuna.exceptions.TrialPruned:
            raise
        except:  # noqa: E722
            return 0

    safe_cn = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in args.cn)
    journal_path = f"./{safe_cn}_optuna.log"
    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=safe_cn,
        load_if_exists=True,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=args.trials)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
