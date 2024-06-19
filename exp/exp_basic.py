import os
from typing import Dict

import torch
from objprint import add_objprint

# removed MambaSimple, Mamba,
from models import (Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer,
                    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer,
                    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, )
from utils.args import Args


@add_objprint(print_methods=True)
class Exp_Basic(object):
    device: torch.device
    model: torch.nn.Module
    model_dict: Dict[str, torch.nn.Module]

    def __init__(self, args: Args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            # 'MambaSimple': MambaSimple,
            # 'Mamba': Mamba,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag: str):
        pass

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def train(self, setting: str) -> 'Exp_Basic':
        """
        train
        @param setting: 用于生成检查点文件路径，无实质作用
        @return:
        """
        pass

    def test(self, setting: str, test=0):
        pass
