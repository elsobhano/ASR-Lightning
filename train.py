import torch
import pytorch_lightning as pl
from model import EnhancemetPipline
from dataLoader import AudioDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("medium") # to make lightning happy

if __name__ == "__main__":
    
    Model = EnhancemetPipline(
        enhacement=None,
        asr_encoder=None,
        stft_layer=None,
        learnig_rate=config.leanring_rate,
    )

    DataModule = AudioDataModule(
        data_dir=config.base_path,
        target_rate=config.SAMPLE_RATE,
        max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.num_wokers
    )

    Trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=3,
        precision=config.PRECISION,
    )

    Trainer.fit(Model, DataModule)
    Trainer.validate(Model, DataModule)
    Trainer.test(Model, DataModule)


