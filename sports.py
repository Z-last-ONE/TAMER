# coding: utf-8


"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import random
import numpy as np
import torch
from utils.quick_start import quick_start
from utils.utils import init_seed

os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    task = 'Sports_and_Outdoors'
    init_seed(999)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='TAMER', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default=task, help='name of datasets')

    config_dict = {
        'gpu_id': 0
    }
    args, _ = parser.parse_known_args()
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
