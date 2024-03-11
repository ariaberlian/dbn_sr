# rbm_sr
conda create -n tf_env python=3.6

conda activate tf_env
pip install -r requirements.txt


### Problem
1. Memory leak: setiap epoch semakin lama
2. Overfitting?? Setiap input keluar output yang sama