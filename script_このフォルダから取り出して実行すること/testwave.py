#シミュレーションで作ったデータのパラメータ推定を行います
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import time
from lal import GreenwichMeanSiderealTime
from astropy.time import Time
from astropy import coordinates as coord
import corner
import os
import gc
import shutil
import h5py
import json
import sys
from sys import exit
from universal_divergence import estimate
import natsort
from vitamin_c import plotting
from tensorflow.keras import regularizers
#import vitamin_c
import vitamin_c.new
import json
from vitamin_c.new import plot_posterior
#from os import *
from vitamin_c import run_vitamin
print(f"DEBUG: Eager execution enabled? {tf.executing_eagerly()}")

params_file = os.path.join(os.path.dirname(__file__), 'vitamin_c', 'params_files', 'params.json')
bounds_file = os.path.join(os.path.dirname(__file__), 'vitamin_c', 'params_files', 'bounds.json')

with open(params_file, 'r') as fp:
    params_dict = json.load(fp)
with open(bounds_file, 'r') as fp:
    bounds = json.load(fp)

samples = run_vitamin.gen_samples(model_loc='20260203_chirp5bai_inverse_model_vitamin_c_run19/model.ckpt.index',
                                  params=os.path.join(os.path.dirname(__file__), 'vitamin_c','params_files','params.json'),
                                  test_set=os.path.join(os.path.dirname(__file__), 'test_sets','1024Hz_full_15par','test_waveforms'),num_samples=100000,
                                  plot_corner=False)

# 1. まず生データをチェック（これが 0 なら、入力データが悪い）
print("=== 1. 生データ (samples) のチェック ===")
print("Shape:", samples.shape)
print("Max:", np.max(samples))
print("Min:", np.min(samples))
print("最初の行:", samples[0]) 
# 生データが死んでいたら(全部0なら)、ここで終了。
if np.max(samples) == 0.0 and np.min(samples) == 0.0:
    print("警告: AIの出力が空っぽ(All Zero)です入力データを確認してください")
    exit() # プログラムを止める

# 3. 【重要】次元を整えて、0-1に「再正規化」する
samples = np.squeeze(samples) # (10000, 15)
norm_samples = np.copy(samples) # コピーを作成

norm_samples = np.zeros_like(samples) # 明示的にゼロで作成

print("\n=== 2. 正規化ループのチェック ===")

inf_pars = params_dict['inf_pars'] # パラメータの名前リスト

for i, param_name in enumerate(inf_pars):
    p_min = bounds[param_name + '_min']
    p_max = bounds[param_name + '_max']
    # 計算してみる
    raw_val = samples[0, i] # 最初のサンプルの値
    norm_val = (raw_val - p_min) / (p_max - p_min)
    
    print(f"[{i}] {param_name}: Raw={raw_val:.2f}, Min={p_min}, Max={p_max} -> Norm={norm_val:.2f}")
    
    # 物理単位 -> 0.0~1.0 に変換
    norm_samples[:, i] = (samples[:, i] - p_min) / (p_max - p_min)

# 3. 最終確認
print("\n=== 3. 最終結果 (norm_samples) ===")
print("Max:", np.max(norm_samples))
print("Samples[0]:", norm_samples[0])

# 4. 正解値（Truth）を作る
# ここで (15,) の配列ができる
x_truth = np.mean(norm_samples, axis=0) 

# 5. プロット実行
idx = 0 # イベントID（適当でOK）

run_dir = time.strftime('test1-%y-%m-%d-%X-%Z')
os.makedirs(run_dir, exist_ok=True)
epoch=1000
print("プロットを開始します...")

# 【修正ポイント】 x_truth[i] ではなく、x_truth をそのまま渡す
plot_posterior(norm_samples, x_truth, epoch, idx,run=run_dir) 

# 保存するファイル名 (compare_results.py で指定したものと合わせる)
# あとで別のサンプルと比較するために、テキストファイルも保存しておくと便利です
save_txt_path = os.path.join(run_dir, f'testwave-posterior_samples_epoch_{epoch}_event_{idx}_vit.txt')

# samples は物理単位になっているので、そのまま保存してOK
np.savetxt(save_txt_path, samples)

print("プロット完了")

