: これは、Anaconda環境でCUDAやcuDNNのライブラリパスを設定するためのバッチファイルです
: GPUを使う場合は、これを実行してからPythonスクリプトを実行してください
: user_nameを実際のユーザー名に置き換えてください
export LD_LIBRARY_PATH=/home/user_name/anaconda3/envs/vitamin39/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/user_name/anaconda3/pkgs/cudnn-8.9.2.26-cuda11_0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/user_name/anaconda3/envs/vitamin39/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH