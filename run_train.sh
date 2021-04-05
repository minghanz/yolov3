######### training

############## To train the network. 

###### Important arguments:
### --tail: using tailed r-box, otherwise using r-box as regression target
### --dual-view: using dual view network, otherwise using bev network only
### --half-angle: using 180-degree range for rotations, otherwise use 360 degrees
### --riou-eval: calculate IOU on the rotated bounding boxes. 
### Default is to set all the above four options. 

### --data: config the training dataset using a .yaml or .data file
### --bev-dataset: if set, the dataset is assumed to be of specific format, with the folders named and organized in a certain way. See dataset_format.txt 

### --tail-inv: using tailed r-box with the tail end as the base, where the feature map is activated. Default off.  
### --img-size: the longer side of input image. Must be multiples of 32. Default: 544. Can choose as the original bev size. 
### --use-mask: mask out regions on bev images where the detections and evaluations are disabled. (KoPER dataset has regions where no gt annotation is available. Therefore the detections there are suppressed by enabling this option.)
### --giou-loss: if set, use iou loss in training, otherwise use other losses. 

python3 train.py \
--epochs 20 --img-size 544 --batch-size 4 --rect --single-cls \
--rotated --rotated-anchor --half-angle --tail --dual-view \
--name ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_nl_stripclr_giout --data data/bev_lturn_CARLA_KoPER_KAligned_texture_mixed_dqueue_shd_nlight.yaml --bev-dataset --riou-eval