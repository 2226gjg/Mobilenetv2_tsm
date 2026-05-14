# resnet50 multi class 
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch resnet50 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=../../../MTK_TSM/checkpoint/imagenet_mean_std_20240924_resnet50_15class_cos_cos/ckpt_epoch_100.pth.tar \
#      --evaluate

# resnet50 multi label
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny.py ucf101 RGB \
#      --arch resnet50 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_multilabel/checkpoint/imagenet_mean_std_20240924_resnet50_multilabel_no_freeze_lr_0.01_cos/ckpt.pth.tar \
#      --evaluate

# mobilenetv2 multi label segment:8
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny_distribution_F1_score.py ucf101 RGB \
     --arch mobilenetv2 --num_segments 8 \
     --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
     --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --lr_type cos\
     --resume=checkpoint/imagenet_mean_std_20240924_mobilenetv2_multilabel_no_freeze_lr_0.01_cos_cos/ckpt.pth.tar \
     --evaluate

# mobilenetv2 multi label segment:16
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv2 --num_segments 16 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=checkpoint/imagenet_mean_std_20240924_mobilenetv2_multilabel_no_freeze_lr_0.05_segement_16_cos_cos/ckpt.pth.tar \
#      --evaluate


# mobilenetv2 multi class
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv2 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=../../../MTK_TSM/checkpoint/imagenet_mean_std_20240924_mobilenetv2_15class_cos_cos/ckpt_epoch_100.pth.tar \
#      --evaluate

