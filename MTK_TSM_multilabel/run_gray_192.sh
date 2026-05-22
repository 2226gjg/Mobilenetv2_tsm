# ------------- mobilenetv2 GRAY 192x192 --------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python3 main.py ucf101 GRAY \
     --arch mobilenetv2 --num_segments 8 \
     --gd 10 --lr 0.01 --lr_steps 10 20 --epochs 300 \
     --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --lr_type cos\
     --input_size 192 \
     --tune_from=pretrained/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth
