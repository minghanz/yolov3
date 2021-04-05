#!/bin/bash
######################### This script is to generate prediction files (no gt, no evaluation)

###### Important arguments:
### --tail (using tailed r-box, otherwise using r-box as regression target)
### --dual-view (using dual view network, otherwise using bev network only)
### --half-angle (using 180-degree range for rotations, otherwise use 360 degrees)
### Default is to set all the above three options. 

### --img-size: the longer side of input image. Must be multiples of 32. Default: 544. Can choose as the original bev size, but not required. 
### --source: path to a folder or a video file. 
### --output: path to a folder. 
### --use-mask: mask out regions on bev images where the detections and evaluations are disabled. 
###             (KoPER dataset has regions where no gt annotation is available. Therefore the detections there are suppressed by enabling this option. BrnoCompSpeed dataset provides similar masks.)

python3 detect.py --rotated --rotated-anchor --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv.pt \
--names data/bev_network.names --single-cls --img-size 448 --save-txt --conf-thres 0.2 --iou-thres 0.2  --use-mask \
--source "" \
--output "" 

# ######### Example on BrnoCompSpeed
# video_name=(
#     # "5_left"
#     # "4_right"
#     # "6_center"
#     "6_right"
# )
# img_size=(
#     # 448
#     # 512
#     # 640
#     512
# )
# for i in ${!video_name[*]}; do 
#     echo "video_name: "${video_name[$i]}

#     python3 detect.py --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout.pt \
#     --names data/bev_network.names --single-cls --img-size ${img_size[$i]} --save-txt --conf-thres 0.3 --iou-thres 0.2  \
#     --source "/media/sda1/datasets/extracted/roadmanship_format/BrnoCompSpeed/session${video_name[$i]}/videos/bev.avi" \
#     --output "/media/sda1/datasets/extracted/roadmanship_format/BrnoCompSpeed/session${video_name[$i]}/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout" #--use-mask
#     # --use-mask
# done