import os
import hydra
import numpy as np
import torch
import torch.nn as nn
from loguru import logger as log
from omegaconf import open_dict
from torch.utils.data import DataLoader
import sys

def init_dataset(cfg):
    '''get the dataloader
    
    return train_loader, val_loader, test_loader, scaler
    '''
    import importlib
    # Dynamic import based on the chosen module
    loader_module = importlib.import_module(f"data_provider.{cfg.dataset_name}")
    get_dataloaders = getattr(loader_module, "get_dataloaders")

    return get_dataloaders(log, cfg)


def init_base(cfg):
    """
    initialize the log path
    """
    # create log path using current time as the file name
    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.add(os.path.join(log_path, "train.log"))

    with open_dict(cfg):
        cfg.log_path = log_path

    log.info(cfg)
    return cfg

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    """
    load model
    """
    if cfg.model_name == 'agcrn':
        from models import AGCRN
        return AGCRN(cfg)
    elif cfg.model_name == 'gwnet':
        from models import GraphWaveNet
        return GraphWaveNet(cfg)
    # elif cfg.model_name == 'deepar':
    #     from models import DeepAR
    #     retrun DeepAR(cfg)
    else:
        raise NotImplementedError

def build_trainer(model, scaler, cfg):
    if cfg.trainer_name == "MTS_trainer":
        from Trainer.MTS_trainer import MTS_trainer
        trainer = MTS_trainer(model, scaler, cfg)
        return trainer
    elif cfg.trainer_name == "MTS_trainer_wEvent":
        from Trainer.MTS_trainer_wEvent import MTS_trainer
        trainer = MTS_trainer(model, scaler, cfg)
        return trainer
    else:
        raise NotImplementedError

@hydra.main(version_base=None, config_path="conf", config_name="config") 
def main(cfg):
    # only for single group name
    group_name = list(cfg.keys())[0]
    cfg = init_base(cfg[group_name])
    seed_everything(cfg.seed)
    
    # dataloader config
    train_loader, valid_loader, test_loader, scaler = init_dataset(cfg)
    
    # model config
    model = build_model(cfg)

    # Trainer config
    trainer = build_trainer(model, scaler, cfg)

    model = trainer.train(train_loader, valid_loader)
    # save weights of the best model,
    torch.save(model.state_dict(), os.path.join(cfg.log_path, "model.pt"))
    metrics, real_y, pred_y = trainer.infer(test_loader)
    if cfg.save_results:
        np.savez(
            os.path.join(cfg.log_path, "result.npz"),
            real_y=real_y,
            pred_y=pred_y,
        )
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
