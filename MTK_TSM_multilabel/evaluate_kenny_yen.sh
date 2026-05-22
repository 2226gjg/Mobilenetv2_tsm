# mobilenetv2 multi class
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv2 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=../../../MTK_TSM/checkpoint/imagenet_mean_std_20240924_mobilenetv2_15class_cos_cos/ckpt_epoch_100.pth.tar \
#      --evaluate

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


# mobilenetv2 multi label with pretrain                                                       碩論有放
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv2 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_multilabel/checkpoint/imagenet_mean_std_20240924_mobilenetv2_multilabel_no_freeze_lr_0.01_cos_cos/ckpt.pth.tar \
#      --evaluate


#mobilenetv2 multi label with out pretrain
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv2 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_multilabel/checkpoint/imagenet_mean_std_20240924_mobilenetv2_multilabel_no_pretrain_no_cos_0.05_cos/ckpt.pth.tar \
#      --evaluate

# # mobilenetv4 multi label Origin UIB first layer 0.05_cos             WOpretrain             碩論有放
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/imagenet_mean_std_20240924_mobilenetv4_multilabel_no_freeze_lr_0.05_cos_cos/ckpt.pth.tar \
#      --evaluate   

# mobilenetv4 multi label Origin UIB first layer 0.05_freeze      WOpretrain
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/imagenet_mean_std_20240924_mobilenetv4_multilabel_no_freeze_lr_0.05_cos/ckpt.pth.tar \
#      --evaluate 

# # mobilenetv4 multi label Before MQA layer 0.05_cos                    WOpretrain                   碩論有放
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/imagenet_mean_std_20240924_mobilenetv4_multilabel_all_residual_no_freeze_lr_0.05_cos_cos/ckpt.pth.tar \
#      --evaluate 

# # mobilenetv4 multi label Origin UIB first layer + MQA layer 0.05_cos     WOpretrain                碩論有放
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/imagenet_mean_std_20240924_mobilenetv4_multilabel_allresidual+UIBfirst_no_freeze_lr_0.05_cos_cos/ckpt.pth.tar \
#      --evaluate 

# mobilenetv4 multi label all UIB + custom UIB 0.05_cos                  WOpretrain                  碩論有放
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/imagenet_mean_std_20240924_mobilenetv4_multilabel_customUIB_no_freeze_lr_0.05_cos_cos/ckpt.pth.tar \
#      --evaluate   

# mobilenetv4 multi label decay 0.01 v2 mAP 0.892                                                       碩論有放
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny.py ucf101 RGB \
     --arch mobilenetv4 --num_segments 8 \
     --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
     --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --lr_type cos\
     --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/imagenet_mean_std_20240924_mobilenetv4_multilabel_customUIB_k400_pretrained_no_freeze_lr_0.01_v2_cos/ckpt.pth.tar \
     --evaluate      

# mobilenetv4 conv_medium                                                       碩論有放
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/20250701_ConvM_lr_0.01_v2_cos/ckpt.pth.tar \
#      --evaluate   

# continuous_frame interval                                                     碩論有放                                                       
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/imagenet_mean_std_20240924_mobilenetv4_multilabel_customUIB_k400_pretrained_no_freeze_lr_0.01_continuousframe_cos/ckpt.pth.tar \
#      --evaluate   

#origin_4_8_12_frame interval                                                       碩論有放
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 300 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/20240924_mobilenetv4_multilabel_customUIB_k400_pretrained_no_freeze_lr_0.01_origin_4_8_12_frame_cos/ckpt.pth.tar \
#      --evaluate   


# Different augmentation                                                        
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 150 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/20240924_HybridM_ColorJitter_RandomErasing_GroupTemporalJitter_RandomHorizontalFlip_cos/ckpt.pth.tar \
#      --evaluate   

#20240924_HybridM_ColorJitter_RandomErasing_cos
#20240924_HybridM_ColorJitter_RandomErasing_GroupTemporalJitter_cos
#20240924_HybridM_ColorJitter_RandomErasing_GroupTemporalJitter_RandomHorizontalFlip_cos
#20240924_HybridM_RandomErasing_cos

# mobilenetv4 conv_Small                                                        碩論有放
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 150 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/20240924_ConvS_lr_0.01_v2_cos/ckpt.pth.tar \
#      --evaluate  

# mobilenetv4 HybridM 20250701                                                 紹任交接用
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_kenny.py ucf101 RGB \
#      --arch mobilenetv4 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 150 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --resume=/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_mobilenetv4_multi/checkpoint/20250701_HybridM_lr_0.01_cos/ckpt.pth.tar \
#      --evaluate  