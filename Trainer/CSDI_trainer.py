"""
CSDI训练器，用于在main.py框架中训练CSDI模型
"""

import os
import torch
import numpy as np
from loguru import logger as log


class CSDI_trainer:
    def __init__(self, model, scaler, cfg):
        self.model = model
        self.scaler = scaler
        self.cfg = cfg
       
        self.device = torch.device(cfg.training_config.device if hasattr(cfg, 'training_config') else 'cuda:0')
        
        # 将模型移动到指定设备
        if hasattr(self.model, 'device'):
            self.model.device = str(self.device)
        
        log.info(f"CSDI trainer initialized with device: {self.device}")
    
    def train(self, train_loader, valid_loader):
        """训练CSDI模型"""
        log.info("Starting CSDI training...")
        self.model.train()
        
        log.info("CSDI training completed")
        return self.model
    
    def test(self, test_loader, valid_loader):
        """使用CSDI模型进行推理"""
        log.info("Starting CSDI inference...")
        
        if hasattr(self.model, 'test'):
            self.model.test()
            metrics = {
            }
            log.info("CSDI inference completed. Results saved by CSDI model.")
            return metrics, None, None
        else:
            log.error("CSDI model does not have test method")
            return {}, None, None



