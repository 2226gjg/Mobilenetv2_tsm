


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py ucf101 RGB \
#      --arch resnet50 --num_segments 8 \
#      --gd 5 --lr 0.001 --lr_steps 10 20 --epochs 100 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth

# CUDA_VISIBLE_DEVICES=3,4,6,7 python3 main.py ucf101 YUV \
#      --arch resnet50 --num_segments 8 \
#      --gd 5 --lr 0.001 --lr_steps 10 20 --epochs 100 \
#      --batch-size 4 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth

# good
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py ucf101 YUV \
#      --arch resnet50 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 100 \
#      --batch-size 16 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth

# CUDA_VISIBLE_DEVICES=2,3,4,7 python3 main.py ucf101 YUV \
#      --arch resnet50 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 100 \
#      --batch-size 16 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth

# # ------------- kenny use ResNet50 --------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python3 main.py ucf101 RGB \
#      --arch resnet50 --num_segments 8 \
#      --gd 10 --lr 0.01 --lr_steps 10 20 --epochs 100 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --tune_from=../../pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth

# ------------- kenny use mobilenetv2 --------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python3 main.py ucf101 GRAY \
     --arch mobilenetv2 --num_segments 8 \
     --gd 10 --lr 0.01 --lr_steps 10 20 --epochs 300 \
     --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --lr_type cos\
     --input_size 256 \
     --tune_from=pretrained/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth


# #experiment on resnet18
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py ucf101 YUV \
#      --arch resnet18 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 25 \
#      --batch-size 4 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\

     
