# Model and data module
from .models import FourierVAE
from ..data import CoronagraphDataModule

# Utils for script automation
import optuna
from lightning.pytorch import Trainer
import torch
from .loss import Loss

ghost_criterion = Loss()

def objective(trial: optuna.trial.Trial):

    dataset = CoronagraphDataModule(12)

    hparams = dict(
    encoder_lr = trial.suggest_float('encoder_lr', 1e-7, 1e-2, log = True),
    encoder_wd = trial.suggest_float('encoder_wd', 1e-7, 1e-2, log = True),
    decoder_lr = trial.suggest_float('decoder_lr', 1e-7, 1e-2, log = True),
    decoder_wd = trial.suggest_float('decoder_wd', 1e-7, 1e-2, log = True),
    optimizer = trial.suggest_categorical('opt', ['adam', 'rms', 'sgd']),
    activation = trial.suggest_categorical('activation', ['relu', 'relu6', 'silu', None]),
    beta = trial.suggest_float('beta', 0.0000001, 1),
    alpha = [
        trial.suggest_float('alpha_1', 0.0000001, 1),
        trial.suggest_float('alpha_2', 0.0000001, 1),
        trial.suggest_float('alpha_3', 0.0000001, 1),
        trial.suggest_float('alpha_4', 0.0000001, 1),
    ]
    )

    model = FourierVAE(**hparams)

    trainer = Trainer(
        logger = True,
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="cuda",
        devices = 1,
        log_every_n_steps=22,
        precision="bf16-mixed",
        limit_train_batches=1 / 3,
        limit_val_batches=1 / 3,
    )

    trainer.fit(model, datamodule=dataset)

    return (
        trainer.callback_metrics["Validation/Pixel"].item(),
        trainer.callback_metrics["Validation/Perceptual"].item(),
        trainer.callback_metrics["Validation/Style"].item(),
        trainer.callback_metrics["Validation/Total variance"].item(),
        trainer.callback_metrics["Validation/KL Divergence"].item(),
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    study = optuna.create_study(directions = ['minimize', 'minimize', 'minimize','minimize', 'minimize'])

    study.optimize(objective, n_trials=150, gc_after_trial=True)

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    for i, name in enumerate(ghost_criterion.labels[:-1]):
        best_param = max(study.best_trials, key = lambda t: t.values[i])
        print(f'Trial with best {name}:')
        print(f"\tnumber: {best_param.number}")
        print(f"\tparams: {best_param.params}")
        print(f"\tvalues: {best_param.values}")
