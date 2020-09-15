#!/bin/sh

# ### using weight trained with KoPER
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_KoPER_all.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4

# ### using weight trained with KoPER
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_KoPER.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_KoPER_all.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_mixed.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt
# python3 test.py --cfg yolov3-spp-rbox.cfg --weights weights/last_gen.pt --data data/bev_KoPERReal.data --conf-thres 0.5 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt

### using CARLA
python3 test.py --data data/bev_KoPERReal.data --conf-thres 0.2 --iou-thres 0.2 --single-cls --rotated --rotated-anchor --img-size 544 --batch-size 4 --save-txt \
--cfg yolov3-spp-rbox180.cfg --half-angle --weights weights/last_KOPER_half_angle.pt