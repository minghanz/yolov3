#!/bin/sh

# ### using weight trained with KoPER
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_KoPER_all.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4

# ### using weight trained with KoPER
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_KoPER.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_KoPER_all.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_mixed.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_gen.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt

# ### using CARLA
# python3 test.py --data data/bev_KoPERReal.data --conf-thres 0.2 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt \
# --cfg yolov3-spp-rbox180.cfg --half-angle --weights weights/last_KOPER_half_angle.pt

# ### rbox180
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rbox180.cfg --rotated --rotated-anchor --half-angle \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_180_l_C_K_KA2000.pt

# ### rbox180 with tail
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_tail_inv_180_l_C_K_KA2000.pt
# # --data data/bev_lturn_CARLA_KoPER_KAligned.yaml --bev-dataset --weights weights/last.pt
# # --data data/bev_KoPERReal.data --weights weights/last_KOPER_half_angle.pt


# ## rbox180 with dual_view (bev-dataset mode)
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rboxdv180.cfg --rotated --rotated-anchor --half-angle --dual-view \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_180_dv__l_C_K_KA2000_ctn21.pt
# # --data data/bev_lturn_CARLA_KoPER_KAligned.yaml --bev-dataset --weights weights/last.pt
# # --data data/bev_KoPERReal.data --weights weights/last_KOPER_half_angle.pt


## rbox360 with tail + dual_view
# echo "----------------------------------------last_ra_360__l_C_K_KA2000"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rbox.cfg --rotated --rotated-anchor \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_360__l_C_K_KA2000.pt
# # --data data/bev_lturn_CARLA_KoPER_KAligned.yaml --bev-dataset --weights weights/last.pt
# # --data data/bev_KoPERReal.data --weights weights/last_KOPER_half_angle.pt


echo "----------------------------------------best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout riou"
python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls  \
--cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --tail --dual-view --half-angle \
--data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout.pt --use-mask --riou

# echo "----------------------------------------best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls  \
# --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --tail --dual-view --half-angle \
# --data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs.pt --use-mask

# echo "----------------------------------------best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs riou"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls  \
# --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --tail --dual-view --half-angle \
# --data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs.pt --use-mask --riou

# echo "----------------------------------------best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls  \
# --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --tail --dual-view --half-angle \
# --data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs.pt --use-mask

# bev_KoPERRealFull.yaml
# echo "----------------------------------------best_ra_180_dv__l_C_K_KA2000_correct_texturedqs"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --use-mask \
# --cfg yolov3-spp-rboxdv180.cfg --rotated --rotated-anchor --dual-view --half-angle \
# --data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_180_dv__l_C_K_KA2000_correct_texturedqs.pt

# echo "----------------------------------------best_ra_tail_180__l_C_K_KA2000_correct_texturedqs"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --use-mask \
# --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --tail --half-angle \
# --data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_tail_180__l_C_K_KA2000_correct_texturedqs.pt

# echo "----------------------------------------best_ra_180__l_C_K_KA2000_correct_texturedqs"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --use-mask \
# --cfg yolov3-spp-rbox180.cfg --rotated --rotated-anchor --half-angle \
# --data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_180__l_C_K_KA2000_correct_texturedqs.pt

# echo "----------------------------------------best_ra_180_dv__l_C_K_KA2000_correct_texturedqs riou"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --use-mask \
# --cfg yolov3-spp-rboxdv180.cfg --rotated --rotated-anchor --dual-view --half-angle \
# --data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_180_dv__l_C_K_KA2000_correct_texturedqs.pt --riou

# echo "----------------------------------------best_ra_tail_180__l_C_K_KA2000_correct_texturedqs riou"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --use-mask \
# --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --tail --half-angle \
# --data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_tail_180__l_C_K_KA2000_correct_texturedqs.pt --riou

# echo "----------------------------------------best_ra_180__l_C_K_KA2000_correct_texturedqs riou"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls --use-mask \
# --cfg yolov3-spp-rbox180.cfg --rotated --rotated-anchor --half-angle \
# --data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_180__l_C_K_KA2000_correct_texturedqs.pt --riou

# echo "----------------------------------------last_ra_tail_360__l_C_K_KA2000_ctn22"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rboxtt.cfg --rotated --rotated-anchor --tail \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_tail_360__l_C_K_KA2000_ctn22.pt

## rbox180 with tail_inv
# echo "----------------------------------------last_ra_tail_inv_180__l_C_K_KA2000"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_tail_inv_180__l_C_K_KA2000.pt

# echo "----------------------------------------last_ra_tail_inv_180__l_C_K_KA2000_ctn22"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_tail_inv_180__l_C_K_KA2000_ctn22.pt

# echo "----------------------------------------last_ra_tail_inv_180__l_C_K_KA2000_ctn22_jump"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rboxtt180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_tail_inv_180__l_C_K_KA2000_ctn22_jump.pt


## rbox180 with tail_inv + dual_view
# echo "----------------------------------------last_ra_tail_inv_180_dv__l_C_K_KA2000"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv --dual-view \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_tail_inv_180_dv__l_C_K_KA2000.pt

# echo "----------------------------------------last_ra_tail_inv_180_dv__l_C_K_KA2000_ctn22"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv --dual-view \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_tail_inv_180_dv__l_C_K_KA2000_ctn22.pt

# echo "----------------------------------------last_ra_tail_inv_180_dv__l_C_K_KA2000_ctn22_jump"
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rboxttdv180.cfg --rotated --rotated-anchor --half-angle --tail --tail-inv --dual-view \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_tail_inv_180_dv__l_C_K_KA2000_ctn22_jump.pt

# ## rbox180
# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rbox180.cfg --rotated --rotated-anchor --half-angle \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_180__l_C_K_KA2000.pt

# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rbox180.cfg --rotated --rotated-anchor --half-angle \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_180__l_C_K_KA2000_ctn22.pt

# python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls \
# --cfg yolov3-spp-rbox180.cfg --rotated --rotated-anchor --half-angle \
# --data data/bev_KoPERReal2.data --bev-dataset --weights weights/last_ra_180__l_C_K_KA2000_ctn22_jump.pt