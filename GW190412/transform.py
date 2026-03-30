import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# --- 設定 ---
gps_time = 1239082262.1  # GW190412の合体時刻(GWOSCを見ること)
duration = 1.0           # 最終的に欲しい長さ
target_fs = 1024         # VItaminの周波数
norm_scale = 16.638832624721797 #ここの数字はparams.jsonのy_normscaleと同じにする

files = {
    'H1': 'H-H1_GWOSC_4KHZ_R1-1239082247-32.hdf5', #ダウンロードしたファイル名を入れる
    'L1': 'L-L1_GWOSC_4KHZ_R1-1239082247-32.hdf5', 
    'V1': 'V-V1_GWOSC_4KHZ_R1-1239082247-32.hdf5'
}

strain_data = []
debug_plots = []

for det in ['H1', 'L1', 'V1']:
    print(f"Processing {det}...")
    
    # 1. まず「前後32秒」だけ読み込む（安全策）
    # ファイル全体を読まず、必要な部分だけ読むことでNaNのリスクを減らす
    # ※ start, end を指定して読む
    try:
        data = TimeSeries.read(files[det], format='hdf5.gwosc', 
                               start=gps_time - 4, end=gps_time + 4)
    except:
        # start/end指定で読めない形式なら全体を読んでcropする
        full_data = TimeSeries.read(files[det], format='hdf5.gwosc')
        data = full_data.crop(gps_time - 16, gps_time + 16)

    # 2. ここでNaNチェック！
    if np.isnan(data.value).any():
        print(f" {det} の生データにNaNが含まれています！補間します。")
        # 欠損値を線形補間で埋める（緊急処置）
        data = data.interpolate(times=data.times)

    # 3. リサンプリング
    data = data.resample(target_fs)
    
    # 4. 白色化 (Whitening)
    # データ端の影響(Edge effect)を避けるため、真ん中を使う
    white_data = data.whiten(4, 2)
    
    # 5. 最終的な切り出し (1秒間)
    # ピークを0.65秒付近に置く
    crop_start = gps_time - 0.65
    crop_end = crop_start + duration
    final_data = white_data.crop(crop_start, crop_end)
    
    # NaN再チェック
    if np.isnan(final_data.value).any():
        print(f" {det}: Whitening後にNaNが発生しました！")
    
    strain_data.append(final_data.value)
    debug_plots.append(final_data) # プロット用に保存

# --- 結合と保存 ---
# (3, 1024) の形にする
strain_stack = np.stack(strain_data, axis=0) 

print(f"作成されたデータの形: {strain_stack.shape}")
print(f"最大値: {np.max(strain_stack)}")
print(f"最小値: {np.min(strain_stack)}")

# 保存
save_name = 'GW190412_event_0.h5py' #ファイル名末尾は数字にすること
with h5py.File(save_name, 'w') as f:
    f.create_dataset('y_data_noisy', data=strain_stack)
    f.create_dataset('y_data_noisefree', data=strain_stack) 
    f.create_dataset('x_data', data=np.zeros((1, 15))) 

print(f"Saved to {save_name}")

# --- ついでに確認プロット ---
plt.figure(figsize=(12, 6))
for i, det in enumerate(['H1', 'L1', 'V1']):
    plt.plot(debug_plots[i].times.value - gps_time, strain_stack[i,:], label=det)
plt.title(f"Processed Waveform around {gps_time}")
plt.xlabel("Time from Merger (s)")
plt.legend()
plt.show()
plt.savefig("check_transform_result.png")
