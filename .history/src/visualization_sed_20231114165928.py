import argparse
import logging
import math
import os
import pickle
import random
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from baseline_utils.ManyHotEncoder import ManyHotEncoder
from dataset import SEDDataset
from loc_vad import activity_detection
from post_processing import EmbedsDataset, ScoreDataset
from trainer import MeanTeacherTrainerOptions
from transforms import get_transforms
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mplfonts import use_font
import seaborn as sb
use_font('Noto Serif CJK SC')#指定中文字体

def seed_everything(seed):
    logging.info("random seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", default="19_proteacher_l3p7", type=str, help="exp name used for the training")
    
    parser.add_argument("--debugmode", default=True, action="store_true", help="Debugmode")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--test_meta", default="/data0/gaolj/sed_data/DCASE2020/metadata/eval/public.tsv")
    parser.add_argument("--test_audio_dir", default="/data0/gaolj/sed_data/DCASE2020/audio/eval/public")
    return parser.parse_args(args)

def extend_arr(arr, scale):
    new_arr = np.zeros((arr.shape[0] * scale, arr.shape[1]))
    for i in range(len(arr)):
        new_arr[i * scale: (i + 1) * scale, :] = arr[i, :]
    return new_arr    

def smooth(data, th_high=0.85, th_low=0.35, n_smooth=3, n_salt=1):
    smoothed_outs = np.zeros((data.shape[0], data.shape[1]))
    for k in range(10):
        bgn_fin_pairs = activity_detection(
            x=data[:, k],
            thres=th_high,
            low_thres=th_low,
            n_smooth=n_smooth,
            n_salt=n_salt)
        for pair in bgn_fin_pairs:
            smoothed_outs[pair[0]:pair[1], k] = data[pair[0]:pair[1], k]
    return smoothed_outs

@torch.no_grad()
def test(exp_name):
    h5_path = "{}/test/test-posterior.h5".format(exp_name)
    h5_path_base = "{}/test/test-posterior-no-p.h5".format(exp_name)
    
    data_loader = DataLoader(ScoreDataset(h5_path , has_label=True))
    dataset_base = ScoreDataset(h5_path_base , has_label=True)
    loc_figs_path = 'exp/sed_figs/dcase_{}/test/'.format(str(exp_name).split('/')[-1][:2])
    labels = ['Speech', 'Dog' , 'Cat', 'Alarm_bell_ringing', 'Dishes', 'Frying', 'Blender', 'Running_water', 'Vacuum_cleaner', 'Electric_shaver_toothbrush']
    event_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    for batch_idx, data in enumerate(data_loader):
        data_id = data["data_id"][0]
        sed_pred = data["pred_strong"].cpu().data.numpy()[0]
        sed_pred_base = dataset_base.dataset[data_id]["pred_strong"]
        target = data["target"].data.numpy()[0]
        fig, axs = plt.subplots(3, 1, figsize=(14, 6), dpi=200, sharex=True)
        # axs[0].set_title(file_name, fontdict={'fontsize': 16, 'family': 'Times New Roman'})
        rm = axs[0].imshow(extend_arr(target.T, 8), cmap='YlOrBr', aspect='auto')
        axs[0].set_yticks(np.arange(4, 10 * 8 + 4, 8))
        axs[0].set_ylabel('真实标签', fontdict={'fontsize': 14, 'family': 'Noto Serif CJK SC'})
        axs[0].set_yticklabels(event_labels, rotation=35, fontdict={'fontsize': 11, 'family': 'Times New Roman'})
        
        axs[1].set_xticks(np.arange(0, 60, 6))
        axs[1].set_xticklabels(np.arange(10.0), fontdict={'fontsize': 14, 'family': 'Times New Roman'})
        axs[1].imshow(extend_arr(sed_pred_base.T, 8), cmap='YlOrBr', aspect='auto')
        axs[1].set_yticks(np.arange(4, 10 * 8 + 4, 8))
        axs[1].set_yticklabels(event_labels, rotation=35, fontdict={'fontsize': 11, 'family': 'Times New Roman'})
        axs[1].set_ylabel('去掉提示令牌', fontdict={'fontsize': 14, 'family': 'Noto Serif CJK SC'})
        axs[2].imshow(extend_arr(sed_pred.T, 8), cmap='YlOrBr', aspect='auto')
        axs[2].set_yticks(np.arange(4, 10 * 8 + 4, 8))
        axs[2].set_yticklabels(event_labels, rotation=35, fontdict={'fontsize': 11, 'family': 'Times New Roman'})
        axs[2].set_ylabel('保留提示令牌', fontdict={'fontsize': 14, 'family': 'Noto Serif CJK SC'})
        
        cb = fig.colorbar(rm, cmap='YlOrBr', ax=axs)
        cb.ax.tick_params(size=14)
        fig.savefig(loc_figs_path + data_id + '.png', bbox_inches='tight')
        plt.close(fig)
       
def test_single(exp_name, id):
    h5_path = "{}/test/test-posterior.h5".format(exp_name)
    h5_path_base = "{}/test/test-posterior-no-p.h5".format(exp_name)
    
    dataset = ScoreDataset(h5_path , has_label=True)
    dataset_base = ScoreDataset(h5_path_base , has_label=True)
    loc_figs_path = 'exp/sed_figs/loc/'.format(str(exp_name).split('/')[-1][:2])
    event_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    data_id = id
    sed_pred = dataset.dataset[data_id]["pred_strong"]
    sed_pred_base = dataset_base.dataset[data_id]["pred_strong"]
    
    th = 0.6
    # sed_pred[sed_pred <= th] = 0.
    # sed_pred_base[sed_pred_base <= th] = 0.
    target = dataset.dataset[data_id]["target"]
    fig, axs = plt.subplots(3, 1, figsize=(14, 6), dpi=200, sharex=True)
    # axs[0].set_title(file_name, fontdict={'fontsize': 16, 'family': 'Times New Roman'})
    rm = axs[0].imshow(extend_arr(target.T, 8), cmap='YlOrBr', aspect='auto')
    axs[0].set_yticks(np.arange(4, 10 * 8 + 4, 8))
    axs[0].set_ylabel('真实标签', fontdict={'fontsize': 16, 'family': 'Noto Serif CJK SC'})
    axs[0].set_yticklabels(event_labels, rotation=35, fontdict={'fontsize': 11, 'family': 'Times New Roman'})
    
    axs[1].set_xticks(np.arange(0, 60, 6))
    axs[1].set_xticklabels(np.arange(10.0), fontdict={'fontsize': 14, 'family': 'Times New Roman'})
    axs[1].imshow(extend_arr((sed_pred_base).T, 8), cmap='YlOrBr', aspect='auto')
    axs[1].set_yticks(np.arange(4, 10 * 8 + 4, 8))
    axs[1].set_yticklabels(event_labels, rotation=35, fontdict={'fontsize': 11, 'family': 'Times New Roman'})
    axs[1].set_ylabel('去掉提示令牌', fontdict={'fontsize': 16, 'family': 'Noto Serif CJK SC'})
    axs[2].imshow(extend_arr(smooth(sed_pred).T, 8), cmap='YlOrBr', aspect='auto')
    axs[2].set_yticks(np.arange(4, 10 * 8 + 4, 8))
    axs[2].set_yticklabels(event_labels, rotation=35, fontdict={'fontsize': 11, 'family': 'Times New Roman'})
    axs[2].set_ylabel('保留提示令牌', fontdict={'fontsize': 16, 'family': 'Noto Serif CJK SC'})
    
    cb = fig.colorbar(rm, cmap='YlOrBr', ax=axs)
    cb.ax.tick_params(size=14)
    fig.savefig(loc_figs_path + data_id + '.png', bbox_inches='tight')
    plt.close(fig)
        
def main(args):
    args = parse_args(args)
    exp_name = Path(f"/home/gaolj/ProTeacher/exp/{args.exp_name}")
    # test(exp_name)
    test_single(exp_name, id='1LKP1ZyHgVg_0_10.wav')
# /YYqaGwN2epw_46_56.wav.png, zM515Ca0AiI_79_89.wav.png, 49AbSai8It4_635_645.wav.png, 3cqsXEzUdiY_248_258.wav.png
if __name__ == "__main__":
    main(sys.argv[1:])
