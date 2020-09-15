#!/bin/sh
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


python3 detect.py --cfg yolov3-spp-rbox180.cfg --weights weights/last_gen_CARLA_half_angle_shadow.pt --names data/bev_network.names --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --half-angle \
--source "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/rain/videos/bev30000.mkv" --output "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/rain/outputs_gen_CARLA_half_angle_shadow/videos"