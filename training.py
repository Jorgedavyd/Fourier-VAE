from framework.trainer.cli import trainer
from data import NormalModule

if __name__ == "__main__":
    trainer(NormalModule, matmul_precision = 'high', deterministic = True, seed = 123)

