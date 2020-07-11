import torch
import numpy as np
import d3d

# boxes = torch.tensor([
#     [10, 10, 3, 5, 0.2],
#     [13, 12, 3, 7, -0.7], 
#     [21, 9, 7, 2, -0.8],
#     [20, 15, 3, 5, 0.6],
#     [18, 5, 7, 4, -0.4],
#     [20, 14, 9, 7, 0.1],
# ], device="cuda:0")
n = 10000
x = torch.rand(n)*200
y = torch.rand(n)*400
w = torch.rand(n)*20 + 10
h = torch.rand(n)*30 + 5
r = torch.rand(n)*2 - 1
boxes1 = torch.stack((x,y,w,h,r), dim=1).to(device="cuda:0")

scores = torch.rand(n).to(device="cuda:0")
# scores = torch.tensor([0.9, 0.3, 0.7, 0.4, 0.5, 0.8], device="cuda:0")

# print(boxes.shape)
# print(scores.shape)
# i_mask = d3d.box.box2d_nms(boxes, scores, iou_method="rbox", iou_threshold=0.3)

n2 = 8000

# x = torch.rand(n2)*200
# y = torch.rand(n2)*400
# w = torch.rand(n2)*20 + 10
# h = torch.rand(n2)*30 + 5
# r = torch.rand(n2)*2 - 1
# boxes2 = torch.stack((x+1e-5,y,w,h,r), dim=1).to(device="cuda:0")

idx = list(range(n2))
# print("idx", idx)
boxes2 = boxes1[idx, :]#.clone().contiguous()
# boxes2 = torch.tensor(boxes1[idx, :])


print(boxes1.shape, boxes1.dtype, boxes1.device)
print(boxes2.shape, boxes2.dtype, boxes2.device)

iou = d3d.box.box2d_iou(boxes1, boxes2, method="rbox") #> iou_thres  # iou matrix
print("iou.shape", iou.shape)
print(iou.max())
print(iou.min())
print(torch.isnan(iou).sum())