cd ../..

python -u run.py \
  --dataset 'Cloud' \
  --root_path ./dataset/ \
  --data ECW.npy \
  --enc_len 48 \
  --pred_len 24 \
  --mpp_update 50 \
  --sim_num 50 \
  --threshold 1 \
  --wave_class 550 \
  --batch_size 256 \
  --freq 'd' \
  --learning_rate 1e-4 \
  --gpu 0