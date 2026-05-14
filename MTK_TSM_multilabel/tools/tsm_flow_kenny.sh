
cd ../
# ------------- kenny use mobilenetv2 --------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=1 python3 main.py ucf101 RGB \
#      --arch mobilenetv2 --num_segments 8 \
#      --gd 10 --lr 0.001 --lr_steps 10 20 --epochs 100 \
#      --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=5 \
#      --shift --shift_div=8 --shift_place=blockres --lr_type cos\
#      --tune_from=pretrained/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth


# RESUME="/data/ivs01/MTK_TSM/checkpoint/imagenet_mean_std_20240924_mobilenetv2_15class_cos_cos/ckpt_epoch_100.pth.tar"
# RESUME="/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_multilabel/checkpoint/imagenet_mean_std_20240924_mobilenetv2_multilabel_no_freeze_lr_0.01_cos_cos/ckpt.pth.tar"
####################### Drama ###############################################
TEST_PATH="/data/ivs01/MTK_TSM/drama/drama_video_label/drama.txt"
RESUME="/data/ivs01/MTK_TSM/checkpoint/imagenet_mean_std_20240924_mobilenetv2_15class_cos_cos/ckpt_epoch_100.pth.tar" ###########################################################################1
REFPATH="/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_multilabel/mtk_dms_20240924/mtk_dms_data_20240924_label/classRef.txt"
OUTPATH="/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_multilabel/mtk_dms_20240924/mtk_dms_data_20240924_mobilenetv2_frame_multiclass_no_text/" ###########################################################################2
OUTPUT_DIR="/data/ivs01/MTK_TSM/TSM_other/MTK_TSM_multilabel/mtk_dms_20240924/mtk_dms_data_20240924_mobilenetv2_video_multiclass_no_text/" ###########################################################################3
python3 visualize_batch_mtk_seatbelt.py --test_path "$TEST_PATH" --resume "$RESUME" --refpath "$REFPATH" --outpath "$OUTPATH" --outputpath "$OUTPUT_DIR"
rm -r $OUTPATH
if [ ! -d "$OUTPATH" ]; then
  mkdir -p "$OUTPATH"
fi
cd tools/
python3 merge_mp4.py "$OUTPUT_DIR/" #合併短影片
rm -r $OUTPATH
cd ../