from lightning.pytorch.cli import LightningCLI
from module.data import CoronagraphDataModule
import torch

if __name__ == "__main__":
    
    torch.set_float32_matmul_precision('high')    

    cli = LightningCLI(
        datamodule_class=CoronagraphDataModule,
        seed_everything_default=123,
        trainer_defaults={
            'deterministic': True, 

        },
    )


