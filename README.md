# rbm_sr
conda create -n tf_env python=3.6

conda activate tf_env
pip install -r requirements.txt


### Problem
1. Memory leak: setiap epoch semakin lama [solved]
2. Setiap input keluar output yang sama. Overfitting?? initiate new model?? [solved](idk why tho)

### Notes
1. Penggunaan beta bodoh. 
Ide awalnya adalah empasis high frequency dengan mengalikannya dengan beta.
Hasilnya berantakan. Error rekonstruksi sangat besar
2. Makin besar epoch makin kecil eror rekons [obvious]
3. Makin besar reps makin kecil psnr+ssim [overfit?]

ariaass
