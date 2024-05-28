from lightorch.htuning.optuna import htuning
from utils import FourierVAE
from data import NormalModule
import optuna
from typing import Dict

def objective(trial: optuna.trial.Trial) -> Dict[str, float]:
    
    return dict(
        encoder_lr = trial.suggest_float('encoder_lr', 1e-4, 1e-1),
        decoder_lr = trial.suggest_float('decoder_lr', 1e-4, 1e-1),
        encoder_wd = trial.suggest_float('encoder_wd', 1e-4, 1e-1),
        decoder_wd = trial.suggest_float('decoder_wd', 1e-4, 1e-1),
        alpha = [
            trial.suggest_float(name, 1e-5, 1, log = True) for name in [
            "Pixel",
            "Perceptual",
            "Style",
            "Total variance"]
        ],
        beta = trial.suggest_float('KL Divergence', 1e-5, 1, log = True),
        
    )
    
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