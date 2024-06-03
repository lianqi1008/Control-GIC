import pytorch_lightning as pl
from omegaconf import OmegaConf
from argparse import ArgumentParser
import torch
import torchvision
import wandb
import numpy as np
import os
from PIL import Image
from CGIC.data.dataset import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from CGIC.util import instantiate_from_config
import CGIC


os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "online"

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    def _wandb(self, pl_module, images, batch_idx, split):
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(wandb.run.dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        formatted_metrics = [f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, float)]
        formatted_metrics_str = "\t".join(formatted_metrics)
        print(f"Epoch {trainer.current_epoch:03d}: {formatted_metrics_str}")

def run_train(config):
    model = instantiate_from_config(config['model'])
    data = LightningDataModule(config['data'])

    if os.path.exists(config['ModelCheckpoint']['dirpath']) == False:
        os.makedirs(config['ModelCheckpoint']['dirpath'])
    callbacks = [
            ModelCheckpoint(**config['ModelCheckpoint']),
            LearningRateMonitor(logging_interval='step'),
            ImageLogger(batch_frequency=1024, max_images=4)  
    ]

    # wandb_logger = WandbLogger(project=config['wandb_project_name'], name='xxx')
    wandb_logger = WandbLogger(project=config['wandb_project_name'])

    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, **config['trainer'])

    if config['ckpt_path'] == '':
        ckpt_path = None
    else:
        ckpt_path = config['ckpt_path']
    trainer.fit(model, data, ckpt_path=ckpt_path)

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='config path')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    run_train(config)