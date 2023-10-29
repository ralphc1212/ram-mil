from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
from glob import glob
import yaml
import torch.nn.functional as F
import csv



parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
					help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="config_template.yaml")
args = parser.parse_args()


def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict

if __name__ == '__main__':
    for i in range(10):
        base_path = 'patch_dir'
        train_val_data = []
        path = 'splits/task_1_tumor_vs_normal_75/' + 'splits_{}.csv'.format(i)
        with open(path,'r') as f:
            reader = csv.DictReader(f)
            train_column = [row1['train'] for row1 in reader]
            for item in train_column:
                if item != '':
                    item = base_path + item + '.pt'
                    train_val_data.append(item)
                    
        with open(path,'r') as f:
            reader = csv.DictReader(f)
            val_column = [row['val'] for row in reader]
            for item in val_column:
                if item != '':
                    item = base_path + item + '.pt'
                    train_val_data.append(item)

        new_feat_path = os.path.join('slide_dir', str(i), 'train')
        if not os.path.exists(new_feat_path):
            os.makedirs(new_feat_path)
        att_path = os.path.join('attention_dir', str(i))
        if not os.path.exists(att_path):
            os.makedirs(att_path)
        config_path = 'heatmaps/configs/config_template.yaml'
        config_dict = yaml.safe_load(open(config_path, 'r'))
        args = config_dict
        model_args = args['model_arguments']
        model_args.update({'n_classes': args['exp_arguments']['n_classes']})
        model_args['ckpt_path'] = 'result/task_1_tumor_vs_normal_CLAM_75_s1/s_'+ str(i) +'_checkpoint.pt'
        model_args['model_type'] = 'clam_sb'
        print(model_args)
        model_args = argparse.Namespace(**model_args)
        ckpt_path = model_args.ckpt_path
        model = initiate_model(model_args, ckpt_path)
        

        emb_files = train_val_data

        for emb_file in emb_files:
            slide_id = os.path.basename(emb_file)
            features = torch.load(emb_file)
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            features = features.to(device)

            logits, Y_prob, Y_hat, A, results_dict = model(features, return_features=True)
            M = results_dict["features"]
            A = F.softmax(A, dim=1) # [1, k]
            torch.save(M, os.path.join(new_feat_path, slide_id))
            torch.save(A, os.path.join(att_path, slide_id))
