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
import os
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


#ディレクトリは波形データが入っているパスを指定してください
#モデルを作ったら，ディレクトリも変えてください
samples = run_vitamin.gen_samples(model_loc='20260203_chirp5bai_inverse_model_vitamin_c_run19/model.ckpt.index',
                                  params=os.path.join(os.path.dirname(__file__), 'vitamin_c','params_files','params.json'),
                                  test_set=os.path.join(os.path.dirname(__file__), 'GW190412'), num_samples=100000,
                                  plot_corner=False)


# 3. 【重要】次元を整えて、0-1に「再正規化」する
samples = np.squeeze(samples) # (10000, 15)
norm_samples = np.copy(samples) # コピーを作成

inf_pars = params_dict['inf_pars'] # パラメータの名前リスト

for i, param_name in enumerate(inf_pars):
    p_min = bounds[param_name + '_min']
    p_max = bounds[param_name + '_max']
    
    # 物理単位 -> 0.0~1.0 に変換
    norm_samples[:, i] = (samples[:, i] - p_min) / (p_max - p_min)

print("DEBUG: Normalized max =", np.max(norm_samples))
print("DEBUG: Normalized min =", np.min(norm_samples)) 

# 4. 正解値（Truth）を作る
# ここで (15,) の配列ができる
x_truth = np.mean(norm_samples, axis=0) 
x_truth[0] = 0.30 #m1の正規化した値です
x_truth[1] = 0.053 #m2を正規化した値です
x_truth[2] = 0.29 #dLを正規化した値です

# 5. プロット実行
idx = 0 # イベントID（適当でOK）

run_dir = time.strftime('GW190412-%y-%m-%d-%X-%Z') #
os.makedirs(run_dir, exist_ok=True)
epoch=20000
print("プロットを開始します...")

plot_posterior(norm_samples, x_truth, epoch, idx,run=run_dir) 

# 保存するファイル名 (compare_results.py で指定したものと合わせる)
# あとで別のサンプルと比較するために、テキストファイルも保存しておくと便利です
save_txt_path = os.path.join(run_dir, f'GW190412-posterior_samples_epoch_{epoch}_event_{idx}_vit.txt')

# samples は物理単位になっているので、そのまま保存してOK
np.savetxt(save_txt_path, samples)

print("プロット完了")

