import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import corner
import os
import json

# ==========================================
# 設定項目
# ==========================================

file = "test0-26-02-05-02-28-25-JST/"

# 1. ファイルパス（実際のファイル名に書き換えてください）
OFFICIAL_SAMPLES_FILE = file + "1024Hz_full_15par_0_samples.dat"       # Bilby/Dynestyの結果
VITAMIN_SAMPLES_FILE = file + "testwave-posterior_samples_epoch_1000_event_0_vit.txt" # VItaminの結果
INJECTION_FILE = file + "1024Hz_full_15par_0.h5py"                     # 正解値 (Truth)
PARAMS_FILE = "vitamin_c/params_files/params.json"                        # 設定ファイル

TRUTH_DICT = {
    # --- 固定したパラメータ ---
    "mass_1": 67.61316114992371,
    "mass_2": 12.85970782774023,
    "luminosity_distance": 1540.352867023132,
    "theta_jn": 1.7910495759004617,
    
    # --- ログを見て書き換える部分 (例: 1.234 など) ---
    "phase": 2.5702790118476058,           # ログの 'phase'
    "ra": 2.2576165768260648,              # ログの 'ra'
    "dec": -0.28275085887262014,             # ログの 'dec'
    "psi": 1.237094944832737,             # ログの 'psi'
    "a_1": 0.40792192767168517,             # ログの 'a_1'
    "a_2": 0.5681183946616588,             # ログの 'a_2'
    "tilt_1": 3.0175821326864596,          # ログの 'tilt_1'
    "tilt_2": 1.43451752131659,          # ログの 'tilt_2'
    "phi_12": 2.6870137537273027,          # ログの 'phi_12'
    "phi_jl": 0.7129134597752139           # ログの 'phi_jl'
}

# ==========================================
# 実行ロジック (Ultimate Range Mode)
# ==========================================

def clean_data(data):
    """NaNやInfを0に置換"""
    if np.any(~np.isfinite(data)):
        # print("Warning: Found NaN/Inf in data. Replacing with 0.")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

def add_jitter(data, epsilon=1e-3):
    """分散がほぼゼロの列にノイズを加える"""
    stds = np.std(data, axis=0)
    zero_std_indices = np.where(stds < 1e-6)[0]
    
    if len(zero_std_indices) > 0:
        print(f"Adding jitter to columns: {zero_std_indices}")
        for idx in zero_std_indices:
            mean_val = np.mean(data[:, idx])
            # ノイズの大きさ: 値の0.1% または epsilon
            scale = max(abs(mean_val) * 1e-3, epsilon)
            noise = np.random.normal(0, scale, size=data.shape[0])
            data[:, idx] += noise
    return data

def main():
    print("--- Loading Configuration (Ultimate Range Mode) ---")

    # 1. パラメータリスト取得
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            params = json.load(f)
        plot_params = params['inf_pars']
    else:
        plot_params = [
            'mass_1', 'mass_2', 'luminosity_distance', 'geocent_time', 'phase', 
            'ra', 'dec', 'psi', 'theta_jn', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'
        ]

    # 2. VItamin読み込み
    samples_vitamin = None
    if os.path.exists(VITAMIN_SAMPLES_FILE):
        try:
            data_vitamin = np.loadtxt(VITAMIN_SAMPLES_FILE)
            if data_vitamin.shape[1] == len(plot_params):
                df_vitamin = pd.DataFrame(data_vitamin, columns=plot_params)
            else:
                df_vitamin = pd.DataFrame(data_vitamin, columns=plot_params[:data_vitamin.shape[1]])
            
            df_vitamin = df_vitamin.fillna(0)
            samples_vitamin = clean_data(df_vitamin.values)
            samples_vitamin = add_jitter(samples_vitamin, epsilon=1e-3)
            
        except Exception as e:
            print(f"VItamin load error: {e}")
            return
    else:
        print("VItamin file not found.")
        return

    # 3. Dynesty読み込み
    samples_official_matrix = None
    if os.path.exists(OFFICIAL_SAMPLES_FILE):
        try:
            df_official_raw = pd.read_csv(OFFICIAL_SAMPLES_FILE, sep='\s+')
            n_official = len(df_official_raw)
            samples_official_matrix = np.zeros((n_official, len(plot_params)))
            
            for i, col in enumerate(plot_params):
                if col in df_official_raw.columns:
                    col_data = df_official_raw[col].values
                    samples_official_matrix[:, i] = col_data
                else:
                    val = TRUTH_DICT.get(col, 0.0)
                    samples_official_matrix[:, i] = val
            
            samples_official_matrix = clean_data(samples_official_matrix)
            samples_official_matrix = add_jitter(samples_official_matrix, epsilon=1e-3)
            
        except Exception as e:
            print(f"Dynesty load error: {e}")

    # 4. 真値リスト
    truths = [TRUTH_DICT.get(p, np.nan) for p in plot_params]

    # 5. プロット範囲の計算 (ここを大改造！)
    # VItamin と Dynesty と Truth の全てを見て、一番広い範囲をとる
    plot_ranges = []
    print("\n--- Calculating Combined Plot Ranges ---")
    
    for i, param in enumerate(plot_params):
        # 比較対象のデータをリストアップ
        candidates = [samples_vitamin[:, i]]
        if samples_official_matrix is not None:
            candidates.append(samples_official_matrix[:, i])
        if truths[i] is not None and np.isfinite(truths[i]):
            # 真値も範囲計算に含める(点として扱う)
            candidates.append(np.array([truths[i]]))

        # 全部つなげて最小・最大を探す
        combined = np.concatenate(candidates)
        v_min, v_max = np.min(combined), np.max(combined)
        
        # 幅の計算
        width = v_max - v_min
        if width < 1e-6: # 幅が狭すぎる場合
            width = max(abs(v_min) * 0.1, 0.1) # 値の10%か、最低0.1の幅を持たせる
            v_avg = (v_min + v_max) / 2
            r_min = v_avg - width
            r_max = v_avg + width
        else:
            # 両側に20%のマージンを持たせる
            r_min = v_min - 0.2 * width
            r_max = v_max + 0.2 * width
            
        plot_ranges.append((r_min, r_max))
        # print(f"  {param}: [{r_min:.2f}, {r_max:.2f}]")

    # ==========================================
    # プロット描画
    # ==========================================
    print("\n--- Generating Plot ---")
    fig = plt.figure(figsize=(20, 20))

    # エラー回避設定: density=Falseにする手もあるが、ヒストグラムが見にくくなるので
    # Combined Rangeを使えば density=True でもいけるはず。
    
    # 1. Dynesty (青) - 先に描画 (後ろ側)
    if samples_official_matrix is not None:
        try:
            corner.corner(
                samples_official_matrix,
                fig=fig,
                labels=plot_params, # ラベルは1回でOKだが念の為
                range=plot_ranges,  # 共通の広い範囲を使う！
                color='tab:blue',
                quantiles=[0.16, 0.84],
                plot_density=False, plot_datapoints=False, fill_contours=True,
                hist_kwargs={'density': True, 'linewidth': 1.5}
            )
        except Exception as e:
            print(f"Dynesty plot skipped: {e}")

    # 2. VItamin (赤) - 後から描画 (手前側)
    try:
        corner.corner(
            samples_vitamin,
            fig=fig,
            labels=plot_params, # ラベル上書き
            range=plot_ranges,  # 共通の広い範囲を使う！
            color='tab:red',
            truths=truths,
            truth_color='tab:orange',
            quantiles=[0.16, 0.84],
            plot_density=False, plot_datapoints=False, fill_contours=True,
            hist_kwargs={'density': True, 'linewidth': 1.5, 'linestyle': '--'}
        )
    except Exception as e:
        print(f"VItamin plot failed: {e}")

    # 凡例
    import matplotlib.lines as mlines
    plt.legend(handles=[
        mlines.Line2D([], [], color='tab:blue', label='Dynesty (Official)'),
        mlines.Line2D([], [], color='tab:red', linestyle='--', label='VItamin (AI)'),
        mlines.Line2D([], [], color='tab:orange', label='Truth')
    ], loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=24, frameon=True)

    output_file = "comp_posterior_final_ultimate.png"
    plt.savefig(output_file)
    print(f"プロット完了: {output_file}")

if __name__ == "__main__":
    main()