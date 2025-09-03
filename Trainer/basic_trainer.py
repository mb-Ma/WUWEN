import os
import torch


class Basic_Trainer(object):
  def __init__(self, model, args):
    self.args = args
    self.device = self._acquire_device()
    self.model = model.to(self.device)
  
  def _acquire_device(self):
    if self.args.use_gpu:
        import platform
        if platform.system() == 'Darwin':
          device = torch.device('mps')
          print('Use MPS')
          return device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
          self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
        device = torch.device('cuda:{}'.format(self.args.gpu))
        if self.args.use_multi_gpu:
          device_ids = args.devices.split(',')
          args.device_ids = [int(id_) for id_ in device_ids]
          print('Use GPU: cuda{}'.format(self.args.device_ids))
        else:
          print('Use GPU: cuda:{}'.format(self.args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device

  def train(self):
    pass

  def infer(self):
    pass