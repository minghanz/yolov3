from utils.utils import plot_one_box, xywh2xyxy_r
import argparse
import bev
import numpy as np
import cv2 
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_result", type=str)
    parser.add_argument("--path_video_bev", type=str)
    parser.add_argument("--path_video_ori", type=str)
    parser.add_argument("--path_video_bev_out", type=str, required=False, default=None)
    parser.add_argument("--path_video_ori_out", type=str, required=False, default=None)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--bspec", type=str)
    parser.add_argument("--calib_path", type=str, required=False, default=None)
    parser.add_argument("--bspec_num", type=float, required=False, default=None)
    parser.add_argument("--calib_new_u", type=int, required=False, default=None)
    parser.add_argument("--calib_new_v", type=int, required=False, default=None)
    parser.add_argument("--frame_start", type=int, required=False, default=0)
    
    args = parser.parse_args()

    with open(args.path_result) as f:
        lines = f.readlines()

    video_ori_in = bev.video_parser(args.path_video_ori, start_id=args.frame_start)
    video_bev_in = bev.video_parser(args.path_video_bev, start_id=args.frame_start)

    dry_run = args.path_video_bev_out is None and args.path_video_ori_out is None
    if not dry_run:
        if args.path_video_ori_out[-3:] in ["mkv", "avi"]:
            out_video_mode = True
        else:
            out_video_mode = False
        if out_video_mode:
            video_ori_out = bev.video_generator(args.path_video_ori_out, dim_from=args.path_video_ori, fps=60)
            video_bev_out = bev.video_generator(args.path_video_bev_out, dim_from=args.path_video_bev, fps=60)
        else:
            os.makedirs(args.path_video_ori_out, exist_ok=True)
            os.makedirs(args.path_video_bev_out, exist_ok=True)
            

    if args.calib in ["lturn", "KoPER", "roundabout"]:
        calib = bev.homo_constr.preset_calib(args.calib, args.bspec_num)
    else:
        assert args.calib in ["CARLA", "blender", "BrnoCompSpeed"]
        calib = bev.homo_constr.load_calib(args.calib, args.calib_path)

    if args.calib_new_u is not None:
        calib = calib.scale(False, new_u=args.calib_new_u, new_v=args.calib_new_v)

    bspec = bev.homo_constr.preset_bspec(args.calib, args.bspec_num, calib)

    H_world_img = calib.gen_H_world_img()
    H_world_bev = bspec.gen_H_world_bev()

    H_img_bev = np.linalg.inv(H_world_img).dot(H_world_bev)

    color = [0, 100, 255]

    frame_cur = args.frame_start - 1
    img_bev = None
    for line in lines:
        if line[0] == "\n":
            continue
        items = line.split(" ")
        if items[-1] == "\n":
            items = items[:-1]
            
        if len(items) == 10:
            mode = "xywhrtt"
        elif len(items) == 8:
            mode = "xywhr"
        else:
            raise ValueError("Not recognized length of line: ({}) {}".format(len(items), items))

        if mode == "xywhrtt":
            frameid, x, y, w, h, r, tx, ty, cls, conf = [int(float(x)) if int(float(x))==float(x) else float(x) for x in items]
        elif mode == "xywhr":
            frameid, x, y, w, h, r, cls, conf = [int(float(x)) if int(float(x))==float(x) else float(x) for x in items]

        if frameid < frame_cur:
            continue

        xywhr = np.array([[x,y,w,h,r]])
        xyxy_8 = xywh2xyxy_r(xywhr)[0]

        xyxy_42 = xyxy_8.reshape((4,2))
        xyxy_43 = np.concatenate((xyxy_42, np.ones((4,1))), axis=1)

        xyxy_ori_43 = H_img_bev.dot(xyxy_43.T).T 
        xyxy_ori_43 = xyxy_ori_43 / xyxy_ori_43[:, [2]]

        xyxy_ori_8 = xyxy_ori_43[:, :2].reshape(-1)
        # print(xyxy_ori_43)

        if mode == "xywhrtt":
            tt = np.array([x+tx, y+ty, 1])
            tt_ori = H_img_bev.dot(tt)
            tt_ori = tt_ori / tt_ori[2]
            tt_offset_ori = tt_ori[:2] - xyxy_ori_43[:, :2].mean(axis=0)
            tt_offset = np.array([tx, ty])

        while frameid > frame_cur:
            if img_bev is not None:
                cv2.imshow("img_bev", img_bev)
                cv2.imshow("img_ori", img_ori)
                cv2.waitKey(1)
                if not dry_run:
                    if out_video_mode:
                        video_bev_out.write(img_bev)
                        video_ori_out.write(img_ori)
                    else:
                        bev_name = os.path.join(args.path_video_bev_out, "{:010d}.jpg".format(frame_cur))
                        ori_name = os.path.join(args.path_video_ori_out, "{:010d}.jpg".format(frame_cur))
                        cv2.imwrite(bev_name, img_bev)
                        cv2.imwrite(ori_name, img_ori)

            img_ori, _ = next(video_ori_in)
            img_bev, _ = next(video_bev_in)
            frame_cur += 1

        # if mode == "xywhrtt":
        #     plot_one_box(np.concatenate([xyxy_8, tt_offset], axis=0), img_bev, label=None, color=color)
        #     plot_one_box(np.concatenate([xyxy_ori_8, tt_offset_ori], axis=0), img_ori, label=None, color=color)
        # elif mode == "xywhr":
        #     plot_one_box(xyxy_8, img_bev, label=None, color=color)
        #     plot_one_box(xyxy_ori_8, img_ori, label=None, color=color)
        if img_bev is not None:
            plot_one_box(xyxy_8, img_bev, label=None, color=color)
            plot_one_box(xyxy_ori_8, img_ori, label=None, color=color)

    if not dry_run:
        if out_video_mode:
            video_bev_out.release()
            video_ori_out.release()