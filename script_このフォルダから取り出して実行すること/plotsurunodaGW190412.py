import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import corner
import os

filenakami = 'GW190412-26-02-03-13-28-21-JST/'
GWevent = 'GW190412'
# --- 設定項目 ---------------------------------------------------
# 1. ダウンロードする(した)公式データのファイル名
OFFICIAL_FILE = filenakami + "IGWN-GWTC2p1-v2-GW190412_053044_PEDataRelease_mixed_cosmo.h5"

# 2. VItaminが出力したサンプルのテキストファイルパス
#    (plotsフォルダの中にある posterior_samples_epoch_XXXX_event_X_vit.txt を指定)
VITAMIN_FILE = filenakami + GWevent + '-posterior_samples_epoch_16000_event_0_vit.txt'

# 3. params.json のパス (パラメータ順序の確認用)
PARAMS_FILE = 'vitamin_c/params_files/params.json'

# 4. プロットしたいパラメータと、公式データ内でのキー名の対応
#    Key: VItaminでの名前, Value: 公式HDF5内での名前
#    ※VItaminがソース質量(source)を学習していると仮定しています
PARAM_MAPPING = {
    'mass_1': 'mass_1_source', 
    'mass_2': 'mass_2_source',
    'luminosity_distance': 'luminosity_distance',
    'theta_jn': 'theta_jn',
    'a_1': 'a_1',
    'a_2': 'a_2',
    'tilt_1': 'tilt_1',
    'tilt_2': 'tilt_2',
    # 必要に応じて追加 (例: 'ra', 'dec' など)
}

# --------------------------------------------------------------

def main():
    # 1. 設定ファイル読み込み
    with open(PARAMS_FILE, 'r') as f:
        params = json.load(f)
    inf_pars = params['inf_pars']  # VItaminが出力するパラメータの順番

    # 2. VItaminデータの読み込み
    print(f"Loading VItamin samples from: {VITAMIN_FILE}")
    vitamin_samples = np.loadtxt(VITAMIN_FILE)
    
    # 3. 公式データの読み込み
    print(f"Loading Official samples from: {OFFICIAL_FILE}")
    official_data = {}
    with h5py.File(OFFICIAL_FILE, 'r') as f:
        # C01:Mixed が推奨波形モデルです
        posterior = f['C01:Mixed']['posterior_samples']
        
        for v_key, o_key in PARAM_MAPPING.items():
            if o_key in posterior.dtype.names:
                official_data[v_key] = posterior[o_key]
            else:
                print(f"Warning: Key '{o_key}' not found in official file.")

    # 4. データの整形（プロット用に配列を揃える）
    # プロットしたいパラメータのインデックスを特定
    plot_indices = []
    plot_labels = []
    official_array = []

    # VItaminのパラメータ順序(inf_pars)に従ってループ
    for i, par in enumerate(inf_pars):
        if par in PARAM_MAPPING and par in official_data:
            plot_indices.append(i)
            # ラベル取得 (params.jsonに定義があれば使う)
            label = params['corner_labels'].get(par, par)
            plot_labels.append(label)
            # 公式データをリストに追加
            official_array.append(official_data[par])

    # 公式データを (N_samples, N_params) の形に変換
    official_samples = np.array(official_array).T
    
    # VItaminデータをフィルタリング
    vitamin_samples_filtered = vitamin_samples[:, plot_indices]

    print(f"Plotting {len(plot_labels)} parameters: {plot_labels}")

# ---------------------------------------------------------
    # 【追加機能】 両方の値を計算してタイトル用の文字列を作る
    # ---------------------------------------------------------
# ---------------------------------------------------------
    # 【修正版】 数値を正しくフォーマットする関数
    # ---------------------------------------------------------
    def get_val_str(samples):
        # 16%, 50%, 84% の分位点を計算
        q_16, q_50, q_84 = np.percentile(samples, [16, 50, 84])
        q_m, q_p = q_50 - q_16, q_84 - q_50
        
        # フォーマット (f文字列を使うのが一番バグが起きにくいです)
        # {:.2f} は「小数点以下2桁」の意味
        label = r"${{{0:.2f}}}_{{-{1:.2f}}}^{{+{2:.2f}}}$".format(q_50, q_m, q_p)
        return label

    # カスタムタイトルのリストを作成
    custom_titles = []
    for i in range(official_samples.shape[1]):
        # 公式の値
        str_off = get_val_str(official_samples[:, i])
        # VItaminの値
        str_vit = get_val_str(vitamin_samples_filtered[:, i])
        
        # 2行にして結合 (青:公式, 赤:VItamin とわかるように色付けも可能だが、まずはテキストで)
        # 改行(\n)を入れることで縦に並べます
        title_text = f"LIGO: {str_off}\nVIt: {str_vit}"
        custom_titles.append(title_text)

    print("Generating corner plot with custom titles...")

    # ---------------------------------------------------------
    # 5. プロット作成 (Corner plot)
    # ---------------------------------------------------------
    
    # まず公式データ（青）
    # titles引数にカスタムリストを渡すのがミソ！
    figure = corner.corner(
        official_samples,
        labels=plot_labels,
        titles=custom_titles, # <--- ここで自作タイトルを指定
        color='tab:blue',
        smooth=0.9,
        quantiles=[0.16, 0.50, 0.84],
        levels=(1-np.exp(-0.5), 1-np.exp(-2.0)),
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        hist_kwargs={'density': True, 'linewidth': 1.5},
        label_kwargs={'fontsize': 14},
        show_titles=False
    )

    # 次にVItaminデータ（赤）を重ねる
    # こちらは show_titles=False にして、タイトルが被らないようにする
    corner.corner(
        vitamin_samples_filtered,
        fig=figure,
        color='tab:red',
        smooth=0.9,
        weights=np.ones(len(vitamin_samples_filtered)) * (len(official_samples) / len(vitamin_samples_filtered)),
        quantiles=[0.16,0.50,0.84],
        levels=(1-np.exp(-0.5), 1-np.exp(-2.0)),
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        hist_kwargs={'density': True, 'linewidth': 1.5, 'linestyle': '--'},
        show_titles=False # <--- 重複回避のためFalse
    )
    # ---------------------------------------------------------
    # 【追加機能】 強制的にタイトルを上書きする（パワープレイ）
    # ---------------------------------------------------------
    # cornerプロットの軸オブジェクトを取得
    ndim = official_samples.shape[1]
    axes = np.array(figure.axes).reshape((ndim, ndim))

    for i in range(ndim):
        # 対角成分（ヒストグラム）のaxを取得
        ax = axes[i, i]
        # カスタムタイトルをセット
        # loc='left' で左寄せにすると、長いタイトルも見やすい
        ax.set_title(custom_titles[i], fontsize=10, loc='left')

    # 凡例の追加
    import matplotlib.lines as mlines
    blue_line = mlines.Line2D([], [], color='tab:blue', label='LIGO Official (GWTC-2.1)')
    red_line = mlines.Line2D([], [], color='tab:red', linestyle='--', label='VItamin (This Work)')
    plt.legend(handles=[blue_line, red_line], loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=14)
    # 保存
    save_path = filenakami + GWevent + '-comparison_.png'
    plt.savefig(save_path,bbox_inches='tight')
    print(f"Done! Plot saved to {save_path}")

if __name__ == '__main__':
    main()