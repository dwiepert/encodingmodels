import subprocess
import os

subjects = ['AA', 'AHfs', 'SJ', 'AL']
root = '/mnt/data/dwiepert/data/features/features_cnk0.1_ctx8.0_pick1_skip5/'
feat_dirs = {'wavlm-large': 'hf/wavlm-large/layer.8', 'ema':'sparc/en', 'lstsq':'lstsq_ema_to_wavlm-large.8', 'pca-lstsq':'pca_lstsq_to_lstsq', 'pca-wav':'pca_wavlm-large.8_to_wavlm-large.8', 'sfa-ema':'emaae_ema_to_sfa-ema'}
save_dir = '/mnt/data/dwiepert/data/corrected_encodingmodels'

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

for s in subjects:
    for ft in feat_dirs:
        fd = os.path.join(root, feat_dirs[ft])
        args = ['python3', 'run_encodingmodels.py', f'--subject={s}', f'--feature_dir={fd}', f'--feature_type={ft}', '--save_dir=/mnt/data/dwiepert/data/corrected_encodingmodels',
            '--sessions', '1', '2', '3', '4', '5', '--nboots=10', '--save_weights', '--save_pred', '--save_crossval' ]

        save_output = os.path.join(save_dir, f'output_{s}_{ft}.txt')

        with open(save_output, 'w') as outfile:
            subprocess.run(args, stdout=outfile, stderr=subprocess.STDOUT)