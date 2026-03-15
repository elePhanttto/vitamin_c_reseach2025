#テストデータを作ります．1つのイベントあたり5時間程かかります
import os
import vitamin_c
from os import *
from vitamin_c import run_vitamin_fix
samples = run_vitamin_fix.gen_test(fixed_vals=os.path.join(os.path.dirname(__file__),'vitamin_c', 'params_files', 'settai.json'))