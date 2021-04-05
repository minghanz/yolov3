#!/bin/sh
############## To generate quantitative evaluation. 

###### Important arguments:
### --tail (using tailed r-box, otherwise using r-box as regression target)
### --dual-view (using dual view network, otherwise using bev network only)
### --half-angle (using 180-degree range for rotations, otherwise use 360 degrees)
### --riou-eval: calculate IOU on the rotated bounding boxes. 
### Default is to set all the above four options. 

### --img-size: the longer side of input image. Must be multiples of 32. Default: 544. Can choose as the original bev size. 
### --use-mask: mask out regions on bev images where the detections and evaluations are disabled. 
###             (KoPER dataset has regions where no gt annotation is available. Therefore the detections there are suppressed by enabling this option. BrnoCompSpeed dataset provides similar masks.)

echo "----------------------------------------best_ra_tail_180_dv"
python3 test.py --img-size 544 --batch-size 4 --save-txt --conf-thres 0.2 --iou-thres 0.2 --single-cls  \
--rotated --rotated-anchor --tail --dual-view --half-angle \
--data data/bev_KoPERRealFull.yaml --bev-dataset --weights weights/best_ra_tail_180_dv.pt --riou-eval --use-mask
