import cv2

import numpy as np

img_path = "/media/sda1/datasets/extracted/shapenet_lturn_3D_black/bev_like_coco/image_vis/blender-000133.color.png"

img = cv2.imread(img_path)

cv2.rectangle(img, (0,int(img.shape[0])), (int(img.shape[1]*0.55), int(img.shape[0]*0.85)), (255,255,255))
cv2.rectangle(img, (0,0), (int(img.shape[1]*0.55), int(img.shape[0]*0.5)), (255,255,255))
cv2.imshow("img", img)
cv2.waitKey()