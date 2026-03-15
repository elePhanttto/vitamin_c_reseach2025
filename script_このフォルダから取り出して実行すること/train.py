import tensorflow as tf
# 他のものをインポートする前に、これを書く！
tf.compat.v1.enable_eager_execution()

import os
import vitamin_c
from os import *
from vitamin_c import run_vitamin
print(f"DEBUG: Eager execution enabled? {tf.executing_eagerly()}")
#run_vitamin.train(resume_training=True)
run_vitamin.train()