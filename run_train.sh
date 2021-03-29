# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --data data/bev_lturn.data --nosave --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

# ######## for testing the work flow
# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name gen5 --data data/bev_gen5.data --single-cls --rotated --rotated-anchor --epochs 3 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

# ######## for training
# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name gen --data data/bev_generalize.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name mixed --data data/bev_mixed.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name KoPER2 --data data/bev_KoPER4000.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

######## finetuning with aligned data
# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name gen_aligned --data data/bev_KoPERAligned.data --single-cls --rotated --rotated-anchor --epochs 5 --batch-size 4 --rect --img-size 544 --weights weights/last_gen.pt

# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name mixed_aligned --data data/bev_KoPERAligned.data --single-cls --rotated --rotated-anchor --epochs 5 --batch-size 4 --rect --img-size 544 --weights weights/last_mixed.pt

########### train with aligned data only: will encounter nan
# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name KoPER_aligned --data data/bev_KoPERAligned.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last_KoPER.pt

# ######### training with aligned and not aligned data
# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name KoPER_all2 --data data/bev_KoPERAll.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

# ######### training with kitti: will encounter nan
# python3 train.py \
# --cfg yolov3-spp-rbox_kitti.cfg --name kitti --data data/bev_kitti.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 512 --bev-dataset #--weights weights/last.pt

# CUDA_LAUNCH_BLOCKING=1 python3 train.py \
# --cfg yolov3-spp-rbox180.cfg --data data/bev.data --nosave --single-cls --rotated --rotated-anchor --epochs 10 --batch-size 4 --rect --img-size 544 --half-angle #--weights weights/last.pt
# python3 train.py --cfg yolov3-spp-nrbox.cfg --data data/bev.data --nosave --single-cls --epochs 10 --batch-size 8 --rect

# ########### training with lturn
# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name lturn --data data/bev_lturn.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 --bev-dataset #--weights weights/last.pt

# ########### training with gen+CARLA
# python3 train.py \
# --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 6 --rect --img-size 544 --bev-dataset \
# --cfg yolov3-spp-rbox180.cfg  --half-angle \
# --name KOPER_half_angle --data data/bev_KoPER.yaml  #--weights weights/last.pt

# ########### training with KOPER_aligned
# python3 train.py \
# --cfg yolov3-spp-rbox180.cfg --name KOPERAligned_half_angle --data data/bev_KoPERAligned.yaml --single-cls --rotated --rotated-anchor --epochs 1 --batch-size 6 --rect --img-size 544 --bev-dataset --half-angle #--weights weights/last.pt

# ######### training without tail or dual-view, 
# python3 train.py \
# --epochs 22 --img-size 544 --batch-size 6 --rect --single-cls \
# --cfg yolov3-spp-rbox180.cfg --rotated --rotated-anchor --half-angle \
# --name ra_180__l_C_K_KA2000_ctn22_jump --data data/bev_KoPER.yaml --bev-dataset --weights weights/last_ra_180__l_C_K_KA2000.pt
# # --name ra_180__l_C_K_KA2000 --data data/bev_lturn_CARLA_KoPER_KAligned.yaml --bev-dataset
# # --name all_tail_half_angle --data data/bev_KoPER/bev_KoPERAligned.yaml #--weights weights/last_all_tail_half_angle_ctn.pt

# ######### training with dual-view, 
# python3 train.py \
# --epochs 22 --img-size 544 --batch-size 6 --rect --single-cls \
# --cfg yolov3-spp-rbox.cfg --rotated --rotated-anchor \
# --name ra_360__l_C_K_KA2000_ctn22 --data data/bev_KoPER.yaml --bev-dataset --weights weights/last_ra_360__l_C_K_KA2000.pt
# # --name ra_360__l_C_K_KA2000 --data data/bev_lturn_CARLA_KoPER_KAligned.yaml --bev-dataset
# # --name ra_tail_360_dv__l_C_K_KA2000_ctn22_jump --data data/bev_KoPER.yaml --bev-dataset --weights weights/last_ra_tail_360_dv__l_C_K_KA2000.pt
# # --name ra_tail_inv_180_dv__l_C_K_KA2000 --data data/bev_lturn_CARLA_KoPER_KAligned.yaml --bev-dataset
# # --name all_tail_half_angle --data data/bev_KoPER/bev_KoPERAligned.yaml #--weights weights/last_all_tail_half_angle_ctn.pt


######### training with tail, 
python3 train.py \
--epochs 20 --img-size 544 --batch-size 4 --rect --single-cls \
--cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --half-angle --tail --dual-view \
--name ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_nl_stripclr_giout --data data/bev_lturn_CARLA_KoPER_KAligned_texture_mixed_dqueue_shd_nlight.yaml --bev-dataset --riou --giou-loss
# --weights weights/last_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giou.pt
# --name ra_tail_180_dv__l_C_K_KA2000_correct_texturedq_ctn22_jump --data data/bev_KoPER_texture_mixed_dqueue.yaml --bev-dataset --weights weights/last_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq.pt
# ra_tail_180_dv__l_C_K_KA2000_correct_texturedq
# ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs
# ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_nl

# --tail-inv
# bev_lturn_CARLA_KoPER_KAligned_texture_mixed
# bev_lturn_CARLA_KoPER_KAligned_texture_mixed_dqueue
# bev_lturn_CARLA_KoPER_KAligned_texture_mixed_dqueue_shd
# bev_lturn_CARLA_KoPER_KAligned_texture_mixed_dqueue_shd_nlight
# bev_KoPER_texture_mixed
# bev_KoPER_texture_mixed_dqueue