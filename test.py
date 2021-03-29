import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True, 
         rotated=False,
         rotated_anchor=False, 
         half_angle=False,
         id=0, 
         output_path=None, 
         bev_dataset=False, 
         tail=False, 
         tail_inv=False, 
         dual_view=False, 
         use_mask=False, 
         riou=False, 
         giou_loss=False):
    ### opt is accessible only when called in this script itself, where opt is declared outside of functions, therefore a global variable
    ### you are also able to change members in opt, since opt is a mutable object. You cannot modify an immutable object as a global variable in a function, except you declare it as global first in the function. 
    ### https://stackoverflow.com/questions/31435603/python-modify-global-list-inside-a-function
    
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('results/test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, imgsz, rotated=rotated, half_angle=half_angle, tail=tail, tail_inv=tail_inv, rotated_anchor=rotated_anchor, dual_view=dual_view)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Fuse
        model.fuse()
        model.to(device)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    if data.endswith("yaml"):
        assert bev_dataset, "yaml data config file can be used only when the dataset is in bev_dataset format"
        data_dict = parse_data_yaml(data)
    else:        
        data_dict = parse_data_cfg(data)
    nc = 1 if single_cls else int(data_dict['classes'])  # number of classes
    path = data_dict['valid']  # path to test images
    names = load_classes(data_dict['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    # iouv = torch.linspace(0.2, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()
    print("niou:", niou)
    print("iouv:", iouv)

    # Dataloader
    if dataloader is None:
        if not dual_view:
            dataset = LoadImagesAndLabels(path, data.rsplit(".", 1)[0]+"_valid", imgsz, batch_size, rect=True, single_cls=single_cls, rotated=rotated, half_angle=half_angle, bev_dataset=bev_dataset, tail=tail, use_mask=use_mask)
            batch_size = min(batch_size, len(dataset))
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                    pin_memory=True,
                                    collate_fn=dataset.collate_fn)
        else:

            dataset_bev = LoadImagesAndLabels(path, data.rsplit(".", 1)[0]+"_valid", imgsz, batch_size, rect=True, single_cls=single_cls, rotated=rotated, half_angle=half_angle, bev_dataset=bev_dataset, tail=tail, dual_view_first=True, use_mask=use_mask)
            
            dataset_ori = LoadImagesAndLabels(path, data.rsplit(".", 1)[0]+"_valid", imgsz, batch_size, rect=True, single_cls=single_cls, rotated=rotated, half_angle=half_angle, bev_dataset=bev_dataset, tail=tail, dual_view_second=True, i_files=dataset_bev.i_files, use_mask=use_mask)
            
            dataset = LoadImagesAndLabelsDual(dataset_bev, dataset_ori)

            batch_size = min(batch_size, len(dataset))
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                    pin_memory=True,
                                    collate_fn=dataset.collate_fn)
            

    seen = 0
    model.eval()
    if not dual_view:
        _ = model(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once
    else:
        _ = model(torch.zeros((1, 3, imgsz, imgsz), device=device), x_sec_view=torch.zeros((1, 3, imgsz, imgsz), device=device), H_img_bev=torch.zeros((1, 3, 3, 3), device=device)) if device.type != 'cpu' else None  # run once

    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    # loss = torch.zeros(3, device=device)
    loss = torch.zeros(7, device=device)       ### to be compatible with rotated bbox, the loss contains 6 terms (adding lxy, lwh, lr, ltt)

    if rotated:
        if tail:
            cls_idx = 8
            conf_idx = 7
        else:
            cls_idx = 6
            conf_idx = 5
    else:
        cls_idx = 5
        conf_idx = 4
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, sample_batch in enumerate(tqdm(dataloader, desc=s)):
        
        if dual_view:
            imgs, invalid_masks, targets, paths, shapes, oris, shapes_oris, H_img_bev = sample_batch
            oris = oris.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            H_img_bev = H_img_bev.to(device).float()
        else:
            imgs, invalid_masks, targets, paths, shapes = sample_batch

        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        invalid_masks = [x.to(device).float() / 255.0 if x is not None else None for x in invalid_masks]

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            if not dual_view:
                inf_out, train_out = model(imgs, augment=augment)  # inference and training outputs
            else:
                inf_out, train_out = model(imgs, augment=augment, x_sec_view=oris, H_img_bev=H_img_bev)  # inference and training outputs

            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                # loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls
                loss += compute_loss(train_out, targets, model, use_giou_loss=giou_loss, half_angle=half_angle, tail_inv=tail_inv)[1][:7]  # GIoU, obj, cls, xy, wh, rotation, tt (rotated or not)

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label, rotated=rotated, rotated_anchor=rotated_anchor, tail=tail, invalid_masks=invalid_masks)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            if not rotated:
                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                if not rotated:
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                else:
                    box = pred[:, :5].clone() # xywhr
                    scale_coords_r(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])

                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[cls_idx])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[conf_idx], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                if not rotated:
                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh # xyxy
                else:
                    tbox = labels[:, 1:6]
                    tbox[:, :4] = tbox[:, :4] * whwh        # xywhr

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, cls_idx]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        if not rotated:
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        else:
                            if riou:
                                # ious, i = d3d.box.box2d_iou(pred[pi, :5], tbox[ti], method="rbox").max(1)
                                pred_d3d = pred[pi, :5].clone()
                                pred_d3d[:,4] = -pred_d3d[:,4]
                                tbox_d3d = tbox[ti].clone()
                                tbox_d3d[:,4] = -tbox_d3d[:,4]
                                ious, i = d3d.box.box2d_iou(pred_d3d, tbox_d3d, method="rbox").max(1)
                            else:
                                ious, i = lin_iou(pred[pi, :5], tbox[ti]).max(1)
                            # print(ious.shape)
                            # print("ious.max()", ious.max())
                            # print("ious.min()", ious.min())
                            

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, conf_idx].cpu(), pred[:, cls_idx].cpu(), tcls))

        # Plot images
        if batch_i < 5:
            f = 'results/test_batch%g_gt.jpg' % batch_i  # filename
            plot_images(imgs, targets, paths=paths, names=names, fname=f)  # ground truth
            f = 'results/test_batch%g_pred%d.jpg' % (batch_i, id)
            if not rotated:
                plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f, gt=False)  # predictions
            else:
                if tail:
                    plot_images(imgs, output_to_target_rtt(output, width, height), paths=paths, names=names, fname=f, gt=False)  # predictions
                else:
                    plot_images(imgs, output_to_target_r(output, width, height), paths=paths, names=names, fname=f, gt=False)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ### above output dim0: w.r.t. class, dim1: w.r.t. iou_threshold
        ### confidence threshold for P,R,F1 is set to a single point, see inside ap_per_class(). AP is irrelevant to confidence threshold since it is the region ratio. 
        # if niou > 1:
        #     p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        #     ### index 0 at dim1 means taking iouv[0] as the iou threshold
        #     ### TODO: notice that above f1 is allocated by ap@0.5 instead of real f1
        # if niou > 1:
        #     p, r, ap, f1 = p[:, 0], r[:, 0], ap[:, 0], f1[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        #     ### index 0 at dim1 means taking iouv[0] as the iou threshold for all thresholds
        if niou > 1:
            # p, r, ap, f1 = p.mean(1), r.mean(1), ap.mean(1), f1.mean(1)  # [P, R, AP@0.5:0.95, AP@0.5]
            mp_iou, mr_iou, map_iou, mf1_iou = p.mean(1), r.mean(1), ap.mean(1), f1.mean(1)  # [P, R, AP@0.5:0.95, AP@0.5]
            ### take average over iou_thresholds
        else:
            mp_iou, mr_iou, map_iou, mf1_iou = p, r, ap, f1

        mp_cls, mr_cls, map_cls, mf1_cls = p.mean(0), r.mean(0), ap.mean(0), f1.mean(0)  # [P, R, AP@0.5:0.95, AP@0.5]
        
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        ### the above mean is over classes
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            # print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
            print(pf % (names[c], seen, nt[c], mp_iou[i], mr_iou[i], map_iou[i], mf1_iou[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    ### save the evaluation result to txt
    if output_path is not None:
        with open(output_path, "a") as f:
            result_dict = {}
            result_dict["PR_conf_thresh(except AP)"] = conf_thres # 0.1 # see ap_per_class class
            result_dict["iou_thresh"] = iouv.tolist()
            result_dict["n_images"] = seen
            result_dict["n_targets"] = nt.sum().item()
            result_dict["mP_cls"] = mp_cls.tolist()
            result_dict["mR_cls"] = mr_cls.tolist()
            result_dict["mF1_cls"] = mf1_cls.tolist()
            result_dict["mAP_cls"] = map_cls.tolist()
            result_dict["mP"] = mp.item()
            result_dict["mR"] = mr.item()
            result_dict["mF1"] = mf1.item()
            result_dict["mAP"] = map.item()
            json.dump(result_dict, f, indent=2)
            f.write("\n")
            speed_dict = {}
            speed_dict["inference"] = t0 *1e3 / seen
            speed_dict["NMS"] = t1 *1e3 / seen
            speed_dict["total"] = (t0+t1) *1e3 / seen
            speed_dict["imgsz_long"] = imgsz
            speed_dict["batch_size"] = batch_size
            json.dump(speed_dict, f, indent=2)
            f.write("\n")
            # print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g\n' % t, file=f)            

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except:
            print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = map_iou[i] # ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--rotated', action='store_true', help='use rotated bbox instead of axis-aligned ones')
    parser.add_argument('--rotated-anchor', action='store_true', help='use residual yaw w.r.t. anchors instead of regressing the original angle')
    parser.add_argument('--half-angle', action='store_true', help='use 180 degree instead of 360 degree in direction estimation (forward-backward equivalent)')
    parser.add_argument('--save-txt', action='store_true', help='save evaluation quantitative results to *.txt')
    parser.add_argument('--bev-dataset', action='store_true', help='use dataset of customized unified structure for bev')
    parser.add_argument('--tail', action='store_true', help='predict tail along with rbox')
    parser.add_argument('--tail-inv', action='store_true', help='predict tail along with rbox, with the xy origin at the end of tail')
    parser.add_argument('--dual-view', action='store_true', help='use both bev and original view as input')
    parser.add_argument('--use-mask', action='store_true', help='use invalid region mask to mask out some regions (do not want detections there)')
    parser.add_argument('--riou', action='store_true', help='use riou threshold in quantitative evaluation')
    parser.add_argument('--giou-loss', action='store_true', help='use giou in loss function')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file
    # opt.data = list(glob.iglob('./**/' + opt.data, recursive=True))[0]  # find file
    print(opt)

    ### save arguments to file
    if opt.save_txt:
        if opt.riou:
            output_path = "results/riou_test_result_d_{}_w_{}.txt".format(opt.data.split("/")[-1].split(".")[0], opt.weights.split("/")[-1].split(".")[0])
        else:
            output_path = "results/liou_test_result_d_{}_w_{}.txt".format(opt.data.split("/")[-1].split(".")[0], opt.weights.split("/")[-1].split(".")[0])
        with open(output_path, "w") as f:
            to_save = opt.__dict__.copy()
            json.dump(to_save, f, indent=2)
            f.write('\n')
    else:
        output_path=None

    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment, 
             rotated=opt.rotated, 
             rotated_anchor=opt.rotated_anchor, 
             output_path=output_path, 
             half_angle=opt.half_angle, 
             bev_dataset=opt.bev_dataset, 
             tail=opt.tail, 
             tail_inv=opt.tail_inv, 
             dual_view=opt.dual_view, 
             use_mask=opt.use_mask, 
             riou=opt.riou, 
             giou_loss=opt.giou_loss)

    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')
