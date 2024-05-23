from framework.htuning.optuna import htuning
from utils import FourierVAE
from data import NormalModule
import optuna
from typing import Dict

def objective(trial: optuna.trial.Trial) -> Dict[str, float]:
    encoder_lr = trial.suggest_float('encoder_lr', 1e-4, 1e-1)
    decoder_lr = trial.suggest_float('decoder_lr', 1e-4, 1e-1)

if __name__ == '__main__':
    htuning(
        model_class = FourierVAE,
        hparam_objective = objective,
        datamodule = NormalModule,
        valid_metrics = [f"Training/{name}" for name in [
            "Pixel",
            "Perceptual",
            "Style",
            "Total variance",
            "KL Divergence"]],
        directions = ['minimize', 'minimize', 'minimize', 'minimize', 'minimize'],
        precision = 'medium',
        n_trials = 150,
    )