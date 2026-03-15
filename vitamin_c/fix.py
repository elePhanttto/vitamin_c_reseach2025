import json
import os
# スペックが低い環境でのトレーニングを安定させるための修正スクリプト
# --- 1. params.json の設定を安全なものに変える ---
params_file = 'params_files/params.json'
print(f"Updating {params_file}...")

with open(params_file, 'r') as f:
    data = json.load(f)

# 保存間隔を短くする（命綱）
data['save_interval'] = 50 
print("  - save_interval: 1000 -> 50")

# ついでに作図間隔も広げてメモリを節約
if data.get('plot_interval', 0) < 100:
    data['plot_interval'] = 100
    print("  - plot_interval -> 100")

# バッチサイズも念のため小さめに維持
if data.get('batch_size', 100) > 32:
    data['batch_size'] = 32
    print("  - batch_size -> 32")

# メモリチャンクも小さく
if data.get('load_chunk_size', 10000) > 2000:
    data['load_chunk_size'] = 2000
    print("  - load_chunk_size -> 2000")

with open(params_file, 'w') as f:
    json.dump(data, f, indent=4)


# --- 2. vitamin_c_new.py にメモリ掃除機能を追加する ---
code_file = 'new.py'
print(f"Patching {code_file}...")

with open(code_file, 'r') as f:
    lines = f.readlines()

new_lines = []
imported_gc = False

for line in lines:
    # 必要なライブラリをインポート
    if "import os" in line and not imported_gc:
        new_lines.append(line)
        new_lines.append("import gc\n") # gc (ゴミ収集) を追加
        imported_gc = True
        continue
    
    # ループの最後（dataset再読み込みの直後あたり）に掃除命令を追加
    # 目印は "train_dataset = (tf.data.Dataset.from_tensor_slices" のブロックの終わり
    new_lines.append(line)
    
    # plot_losses の直後に強制的に図を閉じる命令を入れる
    if "plot_losses(" in line:
        indent = line[:line.find("plot_losses")]
        new_lines.append(f"{indent}plt.close('all')\n")
        new_lines.append(f"{indent}gc.collect()\n")
        new_lines.append(f"{indent}tf.keras.backend.clear_session()\n")
        print("  - Added memory cleanup code after plot_losses")

with open(code_file, 'w') as f:
    f.writelines(new_lines)

print("Done! You are ready to train again.")