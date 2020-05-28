#!/bin/sh
python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.pt --names data/coco_roadusers.names --save-txt --conf-thres 0.6 \
--source "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46.mkv" --output "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46/yolo"