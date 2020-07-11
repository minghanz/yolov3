# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --data data/bev_lturn.data --nosave --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

######## for testing the work flow
python3 train.py \
--cfg yolov3-spp-rbox.cfg --name gen5 --data data/bev_gen5.data --single-cls --rotated --rotated-anchor --epochs 3 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

# ######## for training
# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name gen --data data/bev_generalize.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name mixed --data data/bev_mixed.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name KoPER --data data/bev_KoPER.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

######## finetuning with aligned data
# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name gen_aligned --data data/bev_KoPERAligned.data --single-cls --rotated --rotated-anchor --epochs 5 --batch-size 4 --rect --img-size 544 --weights weights/last_gen.pt

# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name mixed_aligned --data data/bev_KoPERAligned.data --single-cls --rotated --rotated-anchor --epochs 5 --batch-size 4 --rect --img-size 544 --weights weights/last_mixed.pt

# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name KoPER_aligned --data data/bev_KoPERAligned.data --single-cls --rotated --rotated-anchor --epochs 25 --batch-size 4 --rect --img-size 544 --weights weights/last_KoPER.pt

# ######### training with aligned and not aligned data
# python3 train.py \
# --cfg yolov3-spp-rbox.cfg --name KoPER_all --data data/bev_KoPERAll.data --single-cls --rotated --rotated-anchor --epochs 20 --batch-size 4 --rect --img-size 544 #--weights weights/last.pt

# CUDA_LAUNCH_BLOCKING=1 python3 train.py \
# --cfg yolov3-spp-rbox180.cfg --data data/bev.data --nosave --single-cls --rotated --rotated-anchor --epochs 10 --batch-size 4 --rect --img-size 544 --half-angle #--weights weights/last.pt
# python3 train.py --cfg yolov3-spp-nrbox.cfg --data data/bev.data --nosave --single-cls --epochs 10 --batch-size 8 --rect