#!/bin/bash
######################### This script is to generate prediction files (no gt, no evaluation)
########## original yolo object detector
# python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.pt --names data/coco_roadusers.names --save-txt --conf-thres 0.6 \
# --source "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46.mkv" --output "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46/yolo"
# --source "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46/imgs_full" --output "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46/yolo"

########## modified yolo rotated bbox detector
# --weights weights/last_gen.pt       # trained with lturn + KoPER1
# --weights weights/last_KoPER.pt     # trained with KoPER1+4
# --weights weights/last_KoPER_all.pt # trained with KoPER1+4+aligned
# --weights weights/last_mixed.pt     # trained with lturn + KoPER1+4

### using weight trained with lturn
# python3 detect.py --cfg yolov3-spp-rbox.cfg --weights weights/last.pt --names data/bev_network.names --save-txt --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 \
# --source "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist" --output "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist_output_lturn"

# ### using weight trained with gen (KoPER1 + lturn)
# python3 detect.py --cfg yolov3-spp-rbox.cfg --weights weights/last_gen.pt --names data/bev_network.names --save-txt --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 \
# --source "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_4_undist" --output "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_4_undist_output_gen_last"

### using weight trained with KoPER
# python3 detect.py --cfg yolov3-spp-rbox.cfg --weights weights/last_KoPER_all.pt --names data/bev_network.names --save-txt --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 \
# --source "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist/video.avi" --output "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist/outputs/KoPER_all_video2"
# --source "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist/images" --output "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist/outputs/KoPER"

# ### using weight trained with mixed (KoPER + lturn)
# python3 detect.py --cfg yolov3-spp-rbox.cfg --weights weights/last_mixed.pt --names data/bev_network.names --save-txt --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/videos"
# ### using weight trained with mixed (KoPER + lturn+CARLA)
# python3 detect.py --cfg yolov3-spp-rbox.cfg --weights weights/last_gen_CARLA.pt --names data/bev_network.names --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/videos"
# --source "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist" --output "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist_output_mixed"

# --source "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46-bev3000.mkv" --output "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46/yolo-bev20"
# --source "/media/sda1/datasets/extracted/shapenet_lturn_3D_black/bev_like_coco/test" --output "/media/sda1/datasets/extracted/shapenet_lturn_3D_black/bev_like_coco/test_yolo_result_2"


# python3 detect.py --cfg yolov3-spp-rbox.cfg --weights weights/last_gen_CARLA.pt --names data/bev_network.names --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs_gen_CARLA/videos"


# python3 detect.py --cfg yolov3-spp-rbox180.cfg --weights weights/last_gen_CARLA_half_angle_shadow.pt --names data/bev_network.names --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --half-angle \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/rain/videos/bev30000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/rain/outputs_gen_CARLA_half_angle_shadow/videos"

# ### no tail
# python3 detect.py --cfg yolov3-spp-rbox180.cfg --rotated --rotated-anchor --half-angle --weights weights/last_ra_180__l_C_K_KA2000.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_180__l_C_K_KA2000"

# ### no tail
# python3 detect.py --cfg yolov3-spp-rbox180.cfg --rotated --rotated-anchor --half-angle --weights weights/last_ra_180__l_C_K_KA2000_ctn22.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_180__l_C_K_KA2000_ctn22"

# ### tail
# python3 detect.py --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --half-angle --tail --weights weights/last_ra_tail_180__l_C_K_KA2000.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_180__l_C_K_KA2000"

# ### tail
# python3 detect.py --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --half-angle --tail --weights weights/last_ra_tail_180__l_C_K_KA2000_ctn21.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_180__l_C_K_KA2000_ctn21"

# ### dual-view
# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --half-angle --tail --dual-view --weights weights/last_ra_tail_180_dv__l_C_K_KA2000.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_180_dv__l_C_K_KA2000"

# ### dual-view
# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --half-angle --tail --dual-view --weights weights/last_ra_tail_180_dv__l_C_K_KA2000_ctn22.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_180_dv__l_C_K_KA2000_ctn22"


# ### tail-inv
# python3 detect.py --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv --weights weights/last_ra_tail_inv_180__l_C_K_KA2000.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_inv_180__l_C_K_KA2000"

# ### tail-inv
# python3 detect.py --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv --weights weights/last_ra_tail_inv_180__l_C_K_KA2000_ctn22.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_inv_180__l_C_K_KA2000_ctn22"

# ### dual-view
# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv --dual-view --weights weights/last_ra_tail_inv_180_dv__l_C_K_KA2000.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_inv_180_dv__l_C_K_KA2000"

# ### dual-view
# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv --dual-view --weights weights/last_ra_tail_inv_180_dv__l_C_K_KA2000_ctn22.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_inv_180_dv__l_C_K_KA2000_ctn22"


# ########## lturn
# # ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_nl
# # ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs
# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout.pt \
# --names data/bev_network.names --single-cls --img-size 672 --save-txt --conf-thres 0.2 --iou-thres 0.2  \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/night_long/bev_2020-07-14 20-50-17_1hr.mkv" \
# --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout/night_long"
# --use-mask
# night_long/bev_2020-07-14 20-50-17_1hr(1)

# ########## Ko-PER
# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2  \
# --source "/media/sda1/datasets/extracted/roadmanship_format/KoPER_onetake/Sequence1c/KAB_SK_1_undist/videos/bev.avi" \
# --output "/media/sda1/datasets/extracted/roadmanship_format/KoPER_onetake/Sequence1c/KAB_SK_1_undist/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout"
# # --use-mask
########## roundabout
python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_nl.pt \
--names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2  \
--source "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/videos/night_large/bev_10min_4x.mkv" \
--output "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_nl/night_large" --use-mask

# ######### BrnoCompSpeed
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

#     python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout.pt \
#     --names data/bev_network.names --single-cls --img-size ${img_size[$i]} --save-txt --conf-thres 0.3 --iou-thres 0.2  \
#     --source "/media/sda1/datasets/extracted/roadmanship_format/BrnoCompSpeed/session${video_name[$i]}/videos/bev.avi" \
#     --output "/media/sda1/datasets/extracted/roadmanship_format/BrnoCompSpeed/session${video_name[$i]}/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout" #--use-mask
#     # --use-mask
# done


# ### tail 360 dual-view
# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 --use-mask \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/sun/bev_2020-06-30 17-43-28_1hr.mkv" \
# --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq/sun"

# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 --use-mask \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/sun/bev_2020-07-01 19-27-05_1hr.mkv" \
# --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq/sun"

# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 --use-mask \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/sun/bev_2020-10-20 09-31-16_1hr.mkv" \
# --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq/sun"

# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 --use-mask \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/sun/bev_2020-10-27 10-42-37_1hr.mkv" \
# --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq/sun"


# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 --use-mask \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/snow/bev_2020-12-11 09-38-56_1hr.mkv" \
# --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq/snow"

# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 --use-mask \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/snow/bev_2020-12-16 10-36-42_1hr.mkv" \
# --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq/snow"

# python3 detect.py --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor  --half-angle --tail --dual-view --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 --use-mask \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/snow/bev_2020-11-07 20-32-26_1hr30m.mkv" \
# --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq/snow"


# --source "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/night/videos/bev_10min_4x.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/night/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq"
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev30000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq"
# --source "/media/sda1/datasets/extracted/roadmanship_format/KoPER_onetake/Sequence1a/KAB_SK_1_undist/videos/bev.avi" --output "/media/sda1/datasets/extracted/roadmanship_format/KoPER_onetake/Sequence1a/KAB_SK_1_undist/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturetq"
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/sun/bev_2020-07-14 00-49-21_1hr.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/last_ra_tail_180_dv__l_C_K_KA2000_correct_texture"
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/videos/bev_3x.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/outputs/last_ra_tail_180_dv__l_C_K_KA2000_correct_texture"

### tail 360 dual-view
# python3 detect.py --cfg yolov3-spp-rboxttdv.cfg --rotated --rotated-anchor --tail --dual-view --weights weights/last_ra_tail_360_dv__l_C_K_KA2000_ctn22.pt \
# --names data/bev_network.names --single-cls --img-size 544 --save-txt --conf-thres 0.2 --iou-thres 0.2 \
# --source "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_360_dv__l_C_K_KA2000_ctn22"
