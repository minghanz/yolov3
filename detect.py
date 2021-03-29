import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from coco_cvt import cvt2coco

from tqdm import tqdm

def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    rotated = opt.rotated
    rotated_anchor = opt.rotated_anchor
    single_cls = opt.single_cls
    half_angle = opt.half_angle
    tail = opt.tail
    tail_inv = opt.tail_inv
    dual_view = opt.dual_view
    use_mask = opt.use_mask

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    os.makedirs(out, exist_ok=True)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz, rotated=rotated, half_angle=half_angle, tail=tail, tail_inv=tail_inv, rotated_anchor=rotated_anchor, dual_view=dual_view)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        if not dual_view:
            dataset = LoadImages(source, img_size=imgsz, use_mask=use_mask)
        else:
            dataset_bev = LoadImages(source, img_size=imgsz, dual_view_first=True, use_mask=use_mask)
            dataset_ori = LoadImages(source, img_size=imgsz, dual_view_second=True, use_mask=use_mask)
            dataset = LoadImagesDual(dataset_bev, dataset_ori)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    if not single_cls:
        # Get name indices
        name_indices = cvt2coco(opt.names)

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    if not dual_view:
        _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    else:
        H_img_bev = torch.zeros((1, 3, 3, 3), device=device)
        _ = model(img.half() if half else img.float(), x_sec_view=img.half() if half else img.float(), H_img_bev=H_img_bev.half() if half else H_img_bev.float()) if device.type != 'cpu' else None  # run once

    pbar = tqdm(dataset, total=dataset.nframes) # note that nframes is defined only when the source is video
    for frame_i, sample_batch in enumerate(pbar):
    # for path, img, im0s, vid_cap in dataset:
        # if frame_i > 3600:
        #     break
        
        if dual_view:
            path, img, im0s, invalid_mask, vid_cap, H_img_bev, img_ori = sample_batch

            img_ori = torch.from_numpy(img_ori).to(device)
            img_ori = img_ori.half() if half else img_ori.float()  # uint8 to fp16/32
            img_ori /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img_ori.ndimension() == 3:
                img_ori = img_ori.unsqueeze(0)

            H_img_bev = H_img_bev.to(device).float()    # H_img_bev is a torch.Tensor from dataset
            if H_img_bev.ndimension() == 3:
                H_img_bev = H_img_bev.unsqueeze(0)
        else:
            path, img, im0s, invalid_mask, vid_cap = sample_batch

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if invalid_mask is not None:
            invalid_mask = torch.from_numpy(invalid_mask).to(device)
            invalid_mask = invalid_mask.half() if half else invalid_mask.float()  # uint8 to fp16/32
            invalid_mask /= 255.0  # 0 - 255 to 0.0 - 1.0
        invalid_mask = [invalid_mask]

        # Inference
        t1 = torch_utils.time_synchronized()
        if not dual_view:
            pred = model(img, augment=opt.augment)[0]
        else:
            pred = model(img, augment=opt.augment, x_sec_view=img_ori, H_img_bev=H_img_bev)[0]  # inference and training outputs
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms, rotated=rotated, rotated_anchor=rotated_anchor, tail=tail, invalid_masks=invalid_mask)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det_raw in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det_raw is not None and len(det_raw):
                ### preserve both det and det_raw, because H_img_bev converts between det_raw and img_ori
                det = det_raw.clone()

                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if not single_cls:
                    ### the class id (c) are with original coco defs. Therefore we need to map the coco class id to the class id in the input .names file
                    # Print results
                    for c in det[:, -1].unique():
                        if c not in name_indices:
                            continue
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(name_indices[c])])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if not single_cls:
                        if int(cls) not in name_indices:
                            continue
                    if save_txt:  # Write to file
                        if not rotated:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                # file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                                # file.write(('%g ' * 6 + '\n') % (frame_i, cls, *xywh))  # label format with frame_i
                                file.write(('%g ' * 7 + '\n') % (frame_i, *xyxy, cls, conf))  # label format with conf, write xyxy
                        else:
                            xyxy_np = torch.tensor(xyxy).numpy()[None]
                            xywhr_np = xyxy_np[:,:5]
                            xyxy_8 = xywh2xyxy_r(xywhr_np)[0]
                            with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                if not tail:
                                    # file.write(('%g ' * 6 + '\n') % (cls, *xyxy))  # actually xyxy are xywhr
                                    file.write(('%g ' * 8 + '\n') % (frame_i, *xyxy, cls, conf))  # frame_number, xywhr, class, confidence
                                    # file.write(('%g ' * 10 + '\n') % (frame_i, cls, *xyxy_8))  # save point coordinate for easier visualization later
                                else:
                                    file.write(('%g ' * 10 + '\n') % (frame_i, *xyxy, cls, conf))  # frame_number, xywhr, tail_x, tail_y, class, confidence

                    if save_img or view_img:  # Add bbox to image
                        if not rotated:
                            # label = '%s %.2f' % (names[int(name_indices[int(cls)])], conf)
                            label = None
                            plot_one_box(xyxy, im0, label=label, color=colors[int(name_indices[int(cls)])])
                        else:
                            if not single_cls:
                            # if names is not None and len(names) > 1:
                                cls_name = names[int(cls)] if names else int(cls)
                                label = '%s %.1f' % (cls_name, conf)
                            else:
                                label = '%.1f' % conf
                            # label = None
                            if tail:
                                plot_one_box(np.array([ *xyxy_8, xyxy_np[0,5], xyxy_np[0,6] ]), im0, label=label, color=colors[int(cls)])
                            else:
                                plot_one_box(xyxy_8, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))
            pbar.set_description('%sDone. (%.3fs)' % (s, t2 - t1))
            # pbar.update()

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--rotated', action='store_true', help='use rotated bbox instead of axis-aligned ones')
    parser.add_argument('--rotated-anchor', action='store_true', help='use residual yaw w.r.t. anchors instead of regressing the original angle')
    parser.add_argument('--half-angle', action='store_true', help='use 180 degree instead of 360 degree in direction estimation (forward-backward equivalent)')
    parser.add_argument('--tail', action='store_true', help='predict tail along with rbox')
    parser.add_argument('--tail-inv', action='store_true', help='predict tail along with rbox, with the xy origin at the end of tail')
    parser.add_argument('--dual-view', action='store_true', help='use both bev and original view as input')
    parser.add_argument('--use-mask', action='store_true', help='use invalid region mask to mask out some regions (do not want detections there)')
    opt = parser.parse_args()
    opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file
    opt.names = list(glob.iglob('./**/' + opt.names, recursive=True))[0]  # find file
    print(opt)

    with torch.no_grad():
        detect()
