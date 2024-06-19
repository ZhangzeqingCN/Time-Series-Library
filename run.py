import argparse
import random
from typing import Union, Type

import numpy as np
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_basic import Exp_Basic
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.args import print_args, parse_args, Args


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parsed_args: Union[argparse.Namespace, Args] = parse_args()

    # 是否使用GPU

    parsed_args.use_gpu = True if torch.cuda.is_available() and parsed_args.use_gpu else False
    # args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    # 是否使用多GPU

    if parsed_args.use_gpu and parsed_args.use_multi_gpu:
        parsed_args.devices = parsed_args.devices.replace(' ', '')
        device_ids = parsed_args.devices.split(',')
        parsed_args.device_ids = [int(id_) for id_ in device_ids]
        parsed_args.gpu = parsed_args.device_ids[0]

    print('Args in experiment:')
    print_args(parsed_args)

    Exp: Type[Exp_Basic]

    # 按参数选择模型

    if parsed_args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif parsed_args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif parsed_args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif parsed_args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif parsed_args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if parsed_args.is_training:
        for ii in range(parsed_args.itr):
            # setting record of experiments
            exp: Exp_Basic = Exp(parsed_args)  # set experiments
            setting: str = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                parsed_args.task_name,
                parsed_args.model_id,
                parsed_args.model,
                parsed_args.data,
                parsed_args.features,
                parsed_args.seq_len,
                parsed_args.label_len,
                parsed_args.pred_len,
                parsed_args.d_model,
                parsed_args.n_heads,
                parsed_args.e_layers,
                parsed_args.d_layers,
                parsed_args.d_ff,
                parsed_args.expand,
                parsed_args.d_conv,
                parsed_args.factor,
                parsed_args.embed,
                parsed_args.distil,
                parsed_args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            parsed_args.task_name,
            parsed_args.model_id,
            parsed_args.model,
            parsed_args.data,
            parsed_args.features,
            parsed_args.seq_len,
            parsed_args.label_len,
            parsed_args.pred_len,
            parsed_args.d_model,
            parsed_args.n_heads,
            parsed_args.e_layers,
            parsed_args.d_layers,
            parsed_args.d_ff,
            parsed_args.expand,
            parsed_args.d_conv,
            parsed_args.factor,
            parsed_args.embed,
            parsed_args.distil,
            parsed_args.des, ii)

        exp = Exp(parsed_args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    print(F"{args.__dict__=}")
