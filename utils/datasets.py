import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh, xywh2xyxy, xywh2xyxy_r

import re
import bev

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mkv']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


class LoadImagesDual(Dataset):
    def __init__(self, dataset_dual_first, dataset_dual_second):
        self.dual_first = dataset_dual_first
        self.dual_second = dataset_dual_second

        
    def __getattr__(self, name):
        # here you may have to exclude thigs; i.e. forward them to
        # self.name instead for self._impl.name
        # try:
        #     return getattr(self._impl, name)
        # except AttributeError:
        if name == "dual_first":
            return self.dual_first
        elif name == "dual_second":
            return self.dual_second
        else:
            # do something else...
            return getattr(self.dual_first, name)

    def __len__(self):
        return len(self.dual_first)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path_bev, img_bev, img0_bev, invalid_mask, cap_bev, H_world_bev = next(self.dual_first)
        _, img_ori, _, _, _, H_world_img = next(self.dual_second)
        
        # bev, labels_out, path, shapes_bev, H_world_bev = self.dual_first[index]
        # img, _, _, shapes_img, H_world_img = self.dual_second[index]
        
        H_img_world = torch.inverse(H_world_img)
        H_img_bev = torch.matmul(H_img_world, H_world_bev)

        return path_bev, img_bev, img0_bev, invalid_mask, cap_bev, H_img_bev, img_ori

class LoadImages:  # for inference
    def __init__(self, path, img_size=416, dual_view_first=False, dual_view_second=False, use_mask=False):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        self.dual_view_first = dual_view_first                    # dual_view means this dataset is to load original view images. 
        self.dual_view_second = dual_view_second                  # dual_view means this dataset is to load original view images. 

        if self.dual_view_second:
            files = [x.replace("bev", "ori") for x in files]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        print("self.img_size", self.img_size)
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

        self.count = 0

        ### invalid_label_region_masks
        self.use_mask = use_mask
        self.invalid_label_region_mask = [os.path.join(x.split("images")[0], "masks", "bev_invalid_label_region.png") for x in images] + \
                            [os.path.join(x.split("videos")[0], "masks", "bev_invalid_label_region.png") for x in videos]

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        path_invalid_mask = self.invalid_label_region_mask[self.count]
        invalid_mask = None
        if self.use_mask:
            invalid_mask = cv2.imread(path_invalid_mask)
        

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
                    
                    path_invalid_mask = self.invalid_label_region_mask[self.count]
                    if self.use_mask:
                        invalid_mask = cv2.imread(path_invalid_mask)

            self.frame += 1
            # print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            # print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        # img = letterbox(img0, new_shape=self.img_size)[0]
        img, ratio, pad = letterbox(img0, new_shape=self.img_size)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        if invalid_mask is not None:
            invalid_mask, _, _ = letterbox(invalid_mask, new_shape=self.img_size)
            invalid_mask = invalid_mask[:, :, [0]].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            invalid_mask = np.ascontiguousarray(invalid_mask)

        ### for dual_view, preload all calibs and bspecs
        if self.dual_view_first or self.dual_view_second:

            if "CARLA" in path:
                match = re.search("trial_c\d+_\d+", path)
                sub_root = path[:match.end()]
                calib_file = os.path.join(sub_root, 'calibs', 'camera.txt')
                assert os.path.exists(calib_file), calib_file
                calib = bev.homo_constr.load_calib("CARLA", calib_file)
                
                matched_str = path[match.start(): match.end()]
                assert len(matched_str) == 10
                big = int(matched_str[7])
                small = float(matched_str[9])
                code = big + 0.1*small
                bspec = bev.homo_constr.preset_bspec("CARLA", code)
            elif "shapenet_KoPER" in path:
                calib_file = path.replace("images", "labels").replace("ori", "blender").replace("bev", "blender").replace(".png", ".txt")  ### "ori" or "bev" should occur in the path
                assert os.path.exists(calib_file), calib_file
                calib = bev.homo_constr.load_calib("blender", calib_file)

                code = 1 if "shapenet_KoPER1" in path else 4
                bspec = bev.homo_constr.preset_bspec("KoPER", code)
            elif "shapenet_lturn" in path:
                calib_file = path.replace("images", "labels").replace("ori", "blender").replace("bev", "blender").replace(".png", ".txt")
                assert os.path.exists(calib_file), calib_file
                calib = bev.homo_constr.load_calib("blender", calib_file)

                bspec = bev.homo_constr.preset_bspec("lturn")
            elif "KoPER" in path:
                code = 1 if "KAB_SK_1_undist" in path else 4
                calib = bev.homo_constr.preset_calib("KoPER", code)
                bspec = bev.homo_constr.preset_bspec("KoPER", code)
            elif "roundabout" in path:
                calib = bev.homo_constr.preset_calib("roundabout")
                bspec = bev.homo_constr.preset_bspec("roundabout")
            elif "lturn" in path:
                calib = bev.homo_constr.preset_calib("lturn")
                bspec = bev.homo_constr.preset_bspec("lturn")
            elif "BrnoCompSpeed" in path:
                sub_root = path.split("videos")[0]
                calib_file = os.path.join(sub_root, "calibs", "system_dubska_optimal_calib.json")
                video_tag = sub_root.split("session")[-1]
                if video_tag[-1] == "/":
                    video_tag = video_tag[:-1]
                tag_num = int(video_tag[0])
                tag_sub_dict = {"left":0.1, "center":0.2, "right":0.3}
                tag_num = tag_num + tag_sub_dict[video_tag.split("_")[1]]
                calib = bev.homo_constr.load_calib("BrnoCompSpeed", calib_file)
                calib = calib.scale(align_corners=False, new_u=852, new_v=480)
                bspec = bev.homo_constr.preset_bspec("BrnoCompSpeed", tag_num, calib)
            else:
                raise ValueError("file not recognized: {}".format(path))

            if self.dual_view_second:
                assert calib.u_size == img0.shape[1], "calib.u_size: {}, w0: {}".format(calib.u_size, img0.shape[1])
                assert calib.v_size == img0.shape[0], "calib.v_size: {}, h0: {}".format(calib.v_size, img0.shape[0])
                calib = calib.scale(align_corners=False, scale_ratio_u=ratio[0], scale_ratio_v=ratio[1])
                calib = calib.pad(pad[0], pad[1], pad[0], pad[1])
            elif self.dual_view_first:
                assert bspec.u_size == img0.shape[1], "bspec.u_size: {}, w0: {}, img_file: {}".format(bspec.u_size, img0.shape[1], path)
                assert bspec.v_size == img0.shape[0], "bspec.v_size: {}, h0: {}".format(bspec.v_size, img0.shape[0])
                bspec = bspec.scale(align_corners=False, scale_ratio_u=ratio[0], scale_ratio_v=ratio[1])
                bspec = bspec.pad(pad[0], pad[1], pad[0], pad[1])

            if self.dual_view_first:
                H_world_img = bspec.gen_H_world_bev()
                bspec_2 = bspec.scale(align_corners=False, scale_ratio_u=1/8, scale_ratio_v=1/8)
                bspec_4 = bspec_2.scale(align_corners=False, scale_ratio_u=0.5, scale_ratio_v=0.5)
                H_world_img_2 = bspec_2.gen_H_world_bev()
                H_world_img_4 = bspec_4.gen_H_world_bev()
            elif self.dual_view_second:
                H_world_img = calib.gen_H_world_img()
                calib_2 = calib.scale(align_corners=False, scale_ratio_u=1/8, scale_ratio_v=1/8)
                calib_4 = calib_2.scale(align_corners=False, scale_ratio_u=0.5, scale_ratio_v=0.5)
                H_world_img_2 = calib_2.gen_H_world_img()
                H_world_img_4 = calib_4.gen_H_world_img()
            
            H_world_img = torch.from_numpy(H_world_img)
            H_world_img_2 = torch.from_numpy(H_world_img_2)
            H_world_img_4 = torch.from_numpy(H_world_img_4)

            H_stack = torch.stack([H_world_img, H_world_img_2, H_world_img_4], dim=0)

            return path, img, img0, invalid_mask, self.cap, H_stack

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, invalid_mask, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=416):
        self.img_size = img_size

        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=416):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(0 if s == '0' else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

def extract_from_a_path(path):
    path = str(Path(path))  # os-agnostic
    if os.path.isfile(path):  # file
        with open(path, 'r') as f:
            f = f.read().splitlines()
    elif os.path.isdir(path):  # folder
        f = glob.iglob(path + os.sep + '*.*')
    else:
        assert os.path.exists(path), "the path {} does not exist".format(path)
    img_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]

    return img_files

class LoadImagesAndLabelsDual(Dataset):
    def __init__(self, dataset_dual_first, dataset_dual_second):
        self.dual_first = dataset_dual_first
        self.dual_second = dataset_dual_second

        
    def __getattr__(self, name):
        # here you may have to exclude thigs; i.e. forward them to
        # self.name instead for self._impl.name
        # try:
        #     return getattr(self._impl, name)
        # except AttributeError:
        if name == "dual_first":
            return self.dual_first
        elif name == "dual_second":
            return self.dual_second
        else:
            # do something else...
            return getattr(self.dual_first, name)

    def __len__(self):
        return len(self.dual_first)
    
    def __getitem__(self, index):
        bev, invalid_mask, labels_out, path, shapes_bev, H_world_bev = self.dual_first[index]
        img, _, _, _, shapes_img, H_world_img = self.dual_second[index]

        H_img_world = torch.inverse(H_world_img)
        H_img_bev = torch.matmul(H_img_world, H_world_bev)

        if self.dual_first.augment:
            img, img_np, img_blank = augment_strip_occlusion(img, torch_mode=True)

            H_img_bev_cur_np = H_img_bev[0].numpy() # 3*3
            H_bev_img_cur_np = np.linalg.inv(H_img_bev_cur_np)

            bev_img = cv2.warpPerspective(img_np, H_bev_img_cur_np, (bev.shape[2],bev.shape[1]))
            bev_blank = cv2.warpPerspective(img_blank, H_bev_img_cur_np, (bev.shape[2],bev.shape[1]))

            bev, _, _ = augment_strip_occlusion(bev, torch_mode=True, img_blank=bev_blank, img_src=bev_img)            

            bev, dropped_channels = augment_channel_drop(bev, torch_mode=True)
            img, _ = augment_channel_drop(img, torch_mode=True, dropout_chnl=dropped_channels)
        
        return bev, invalid_mask, labels_out, path, shapes_bev, img, shapes_img, H_img_bev

    @staticmethod
    def collate_fn(batch):
        bev, invalid_mask, label, path, shapes_bev, img, shapes_img, H_img_bev = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(bev, 0), invalid_mask, torch.cat(label, 0), path, shapes_bev, torch.stack(img, 0), shapes_img, torch.stack(H_img_bev, 0)

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, path_cache, img_size=416, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, rotated=False, half_angle=False, bev_dataset=False, tail=False, dual_view_first=False, dual_view_second=False, i_files=None, use_mask=False):
        try:
            if isinstance(path, list):
                self.img_files = []
                for path_i in path:
                    img_files = extract_from_a_path(path_i)
                    self.img_files.extend(img_files)
            else:
                self.img_files = extract_from_a_path(path)
            self.img_files = sorted(self.img_files)
        except:
            raise Exception('Error loading data from %s. See %s' % (path, help_url))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.rotated = rotated
        self.half_angle = half_angle
        self.tail = tail
        self.dual_view_first = dual_view_first                    # dual_view means this dataset is to load original view images. 
        self.dual_view_second = dual_view_second                  # dual_view means this dataset is to load original view images. 
        self.use_mask = use_mask
        
        ### for rotated bbox training, turn on rect to disable mosaic training!!!, set the long side using img_size, and short side will be padded to min 32x

        # Define labels
        if bev_dataset:
            if tail:
                self.label_files = [x.replace('images', 'labels').replace('bev', 'rboxtt_coco').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]
                if not all(os.path.exists(x) for x in self.label_files):
                    print("rbox_tt labeling folder does not exist, turning to rbox_coco folder instead, then adding dummy tail labels")
                    for x in self.label_files:
                        if not os.path.exists(x):
                            print("Rboxtt not existing:", x)
                            break
                    self.label_files = [x.replace('images', 'labels').replace('bev', 'rbox_coco').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]
            else:
                self.label_files = [x.replace('images', 'labels').replace('bev', 'rbox_coco').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]
        else:
            self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        if self.dual_view_second:
            self.img_files = [x.replace("bev", "ori") for x in self.img_files]

        if self.dual_view_first:
            print(all("ori" not in imgfile for imgfile in self.img_files))
            print(self.dual_view_second)

        ### bev mask for valid labels region:   # the path may not exist, in which case the whole image is valid
        self.invalid_label_region_mask = [os.path.join(x.split("images")[0], "masks", "bev_invalid_label_region.png") for x in self.img_files]

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        ### Minghan: this section enables training using images of different sizes. 
        # All images are sorted according to aspect ratio, so that images in the same batch should have similar aspect ratio. 
        # Then each mini-batch is assigned with a common batch-shape, such that all shaped in this batch can be fit into it. 
        if self.rect:
            # Read image shapes (wh)
            # if isinstance(path, list):
            #     sp = path[0].replace('.txt', '.shapes')  # shapefile path
            # else:
            #     sp = path.replace('.txt', '.shapes')  # shapefile path
            sp = path_cache+'.shapes'
            try:
                with open(sp, 'r') as f:  # read existing shapefile
                    print("reading existing shapefile at:", os.path.abspath(sp))
                    s = [x.split() for x in f.read().splitlines()]
                    assert len(s) == n, 'Shapefile out of sync, {} {}'.format(len(s), n)
            except:
                s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
                np.savetxt(sp, s, fmt='%g')  # overwrites existing (if any)

            # Sort by aspect ratio
            s = np.array(s, dtype=np.float64)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            if self.dual_view_second:
                assert i_files is not None, "dual_view_second mode should use the file sequences generated by dual_view_first. "
                i = i_files
            elif self.dual_view_first:
                self.i_files = i                 ## save the sequence of files for the initialization of the original view dataset (dual_view_second)
            ### when the aspect ratio is identical for all images, this may produce unexpected sorting result
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            self.invalid_label_region_mask = [self.invalid_label_region_mask[i] for i in i]
            self.shapes = s[i]  # wh
            ar = ar[i]
            # self.shapes = s

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            # self.batch_shapes = np.ceil(np.array(shapes) * img_size / 64.).astype(np.int) * 64
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.).astype(np.int) * 32 ### see gs in train.py

        # Cache labels
        self.imgs = [None] * n
        if self.rotated:
            if self.tail:
                self.labels = [np.zeros((0, 8), dtype=np.float32)] * n
            else:
                self.labels = [np.zeros((0, 6), dtype=np.float32)] * n
        else:
            self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        # np_labels_path = str(Path(self.label_files[0]).parent) + '.npy'  # saved labels in *.npy file
        np_labels_path = path_cache + '.npy'
        if os.path.isfile(np_labels_path):
            print("Existing: np_labels_path", np_labels_path)
            s = np_labels_path
            x = list(np.load(np_labels_path, allow_pickle=True))
            if len(x) == n:
                self.labels = x
                labels_loaded = True
        else:
            # if isinstance(path, list):
            #     s = path[0].replace('images', 'labels')
            # else:
            #     s = path.replace('images', 'labels')
            # print("path", path)
            s = np_labels_path
            print("Non-existing: np_labels_path", np_labels_path)

        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            if labels_loaded:
                l = self.labels[i]
            else:
                try:
                    with open(file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                    continue

            if l.size:
                if self.rotated:
                    if self.tail:
                        assert l.shape[1] in [6,8], '%d label columns while 8 are required: %s' % (l.shape[1], file)
                        assert (l[:,:5] >= 0).all(), 'negative labels: %s' % file
                        assert (l[:, 1:5] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                        if l.shape[1] == 6:
                            # print("tail gt is missing, appending dummy zeros")
                            l = np.concatenate((l, np.zeros((l.shape[0], 2))), axis=1)

                        # ### old checking: removes labels with too long tail
                        # if not (np.abs(l[:, [6,7]]) <= 1).all():
                        #     print('non-normalized or out of bounds coordinate labels: %s' % file)
                        #     iir = []
                        #     for ir in range(l.shape[0]):
                        #         if (np.abs(l[ir, [6,7]]) <= 1).all():
                        #             iir.append(ir)
                        #     l = l[iir]

                        ### new checking: 11282020
                        ### 1. if tail is in the wrong direction (dy should always < 0), go opposite direction. 
                        if not (l[:, 7] <= 0).all():
                            print('tail end at region where perspective transform is not defined: %s' % file)
                            for ir in range(l.shape[0]):
                                if l[ir, 7] > 0:
                                    l[ir, 6:8] = - l[ir, 6:8]

                        ### 2. clip tail end to zeros. 
                        tail_end = l[:, 1:3] + l[:, 6:8]
                        tail_end = tail_end.clip(0, 1)
                        l[:, 6:8] = tail_end - l[:, 1:3]

                        l[:,1:3] = l[:,1:3].clip(1e-5) ### values very close to zero may cause numeric issue (7e-17 for example)
                    else:
                        assert l.shape[1] in [5,6], '%d label columns while 6 are required: %s' % (l.shape[1], file)
                        assert (l[:,:5] >= 0).all(), 'negative labels: %s' % file
                        assert (l[:, 1:5] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                        if l.shape[1] == 5:
                            l = np.concatenate((l, np.zeros((l.shape[0], 1))), axis=1) ## add a virtual yaw dim
                        l[:,1:3] = l[:,1:3].clip(1e-5) ### values very close to zero may cause numeric issue (7e-17 for example)
                else:
                    assert l.shape[1] in [5, 6], '%d label columns while 5 are required: %s' % (l.shape[1], file)
                    assert (l[:, :5] >= 0).all(), 'negative labels: %s' % file
                    assert (l[:, 1:5] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                    if l.shape[1] == 6:
                        l = l[:, :5] ## delete the yaw dim

                if l.shape[0]:
                    if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                        nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                    if single_cls:
                        l[:, 0] = 0  # force dataset into single-class mode
                    if self.rotated and self.half_angle:    
                        ### so that the gt angle is between 0 and pi (later there may be augmentation so it could still go out of the range, but it's fine) 
                        l[l[:, 5]<0, 5] += math.pi

                    self.labels[i] = l
                    nf += 1  # file found

                    # Create subdataset (a smaller dataset)
                    if create_datasubset and ns < 1E4:
                        if ns == 0:
                            create_folder(path='./datasubset')
                            os.makedirs('./datasubset/images')
                        exclude_classes = 43
                        if exclude_classes not in l[:, 0]:
                            ns += 1
                            # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                            with open('./datasubset/images.txt', 'a') as f:
                                f.write(self.img_files[i] + '\n')

                    # Extract object detection boxes for a second stage classifier
                    if extract_bounding_boxes:
                        p = Path(self.img_files[i])
                        img = cv2.imread(str(p))
                        h, w = img.shape[:2]
                        for j, x in enumerate(l):
                            f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(Path(f).parent)  # make new output folder

                            if self.rotated:
                            ### I guess the labels are: [class, x, y, width, height, yaw] from below
                                b = x[1:6]
                                b[:4] = b[:4] * [w, h, w, h]  # box
                                b[2:4] = b[2:4].max()  # rectangle to square
                                b[2:4] = b[2:4] * 1.3 + 30  # pad
                                b = xywh2xyxy_r(b.reshape(-1, 5), external_aa=True).ravel().astype(np.int)
                            else:
                            ### I guess the labels are: [class, x, y, width, height] from below
                                b = x[1:] * [w, h, w, h]  # box
                                b[2:] = b[2:].max()  # rectangle to square
                                b[2:] = b[2:] * 1.3 + 30  # pad
                                b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
                                ### b only has one dim, so reshape is fine, otherwise should use transpose?

                            b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                            assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
                else:
                    ne += 1
            
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = 'Caching labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                s, nf, nm, ne, nd, n)
        assert nf > 0, 'No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)
        if not labels_loaded:
            print('Saving labels to %s for faster future loading' % np_labels_path)
            np.save(np_labels_path, self.labels)  # save for next time

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)
            if self.dual_view_second:
                invalid_mask = None
            else:
                invalid_mask, hw0_mask, hw_mask = load_image_mask(self, index)
                if invalid_mask is not None:
                    h0_mask, w0_mask = hw0_mask
                    h_mask, w_mask = hw_mask
                    assert h == h_mask
                    assert w == w_mask

            # print("-------------")
            # print("VIS img.shape", img.shape, self.dual_view_first)
            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            if invalid_mask is not None:
                invalid_mask, _, _ = letterbox(invalid_mask, shape, auto=False, scaleup=self.augment)

            # print("VIS img.shape", img.shape, self.dual_view_first)
            ### load bev spec
            ### for dual_view, load corresponding bspec and calib
            ### modify it according to the scaling and padding

            ### for dual_view, preload all calibs and bspecs
            if self.dual_view_first or self.dual_view_second:

                if "CARLA" in self.img_files[index]:
                    match = re.search("trial_c\d+_\d+", self.img_files[index])
                    sub_root = self.img_files[index][:match.end()]
                    calib_file = os.path.join(sub_root, 'calibs', 'camera.txt')
                    assert os.path.exists(calib_file), calib_file
                    calib_0 = bev.homo_constr.load_calib("CARLA", calib_file)
                    
                    matched_str = self.img_files[index][match.start(): match.end()]
                    assert len(matched_str) == 10
                    big = int(matched_str[7])
                    small = float(matched_str[9])
                    code = big + 0.1*small
                    bspec_0 = bev.homo_constr.preset_bspec("CARLA", code)
                elif "shapenet_KoPER" in self.img_files[index]:
                    calib_file = self.img_files[index].replace("images", "labels").replace("ori", "blender").replace("bev", "blender").replace(".png", ".txt")  ### "ori" or "bev" should occur in the path
                    assert os.path.exists(calib_file), calib_file
                    calib_0 = bev.homo_constr.load_calib("blender", calib_file)

                    code = 1 if "shapenet_KoPER1" in self.img_files[index] else 4
                    bspec_0 = bev.homo_constr.preset_bspec("KoPER", code)
                elif "shapenet_lturn" in self.img_files[index]:
                    calib_file = self.img_files[index].replace("images", "labels").replace("ori", "blender").replace("bev", "blender").replace(".png", ".txt")
                    assert os.path.exists(calib_file), calib_file
                    calib_0 = bev.homo_constr.load_calib("blender", calib_file)

                    bspec_0 = bev.homo_constr.preset_bspec("lturn", 0)  ## synthetic lturn, use original bspec
                elif "KoPER" in self.img_files[index]:
                    code = 1 if "KAB_SK_1_undist" in self.img_files[index] else 4
                    calib_0 = bev.homo_constr.preset_calib("KoPER", code)
                    bspec_0 = bev.homo_constr.preset_bspec("KoPER", code)
                elif "roundabout" in self.img_files[index]:
                    calib_0 = bev.homo_constr.preset_calib("roundabout")
                    bspec_0 = bev.homo_constr.preset_bspec("roundabout")
                elif "lturn" in self.img_files[index]:      ## real lturn, 
                    calib_0 = bev.homo_constr.preset_calib("lturn")
                    bspec_0 = bev.homo_constr.preset_bspec("lturn")
                else:
                    raise ValueError("file not recognized: {}".format(self.img_files[index]))

                if self.dual_view_second:
                    assert calib_0.u_size == w0, "calib_0.u_size: {}, w0: {}".format(calib_0.u_size, w0)
                    assert calib_0.v_size == h0, "calib_0.v_size: {}, h0: {}".format(calib_0.v_size, h0)
                    calib = calib_0.scale(align_corners=False, new_u=w, new_v=h)
                    calib = calib.pad(pad[0], pad[1], pad[0], pad[1])
                elif self.dual_view_first:
                    assert bspec_0.u_size == w0, "bspec_0.u_size: {}, w0: {}, img_file: {}".format(bspec_0.u_size, w0, self.img_files[index])
                    assert bspec_0.v_size == h0, "bspec_0.v_size: {}, h0: {}".format(bspec_0.v_size, h0)
                    bspec = bspec_0.scale(align_corners=False, new_u=w, new_v=h)
                    bspec = bspec.pad(pad[0], pad[1], pad[0], pad[1])

                # print("VIS h0 w0 h w pad", h0, w0, h, w, pad, self.dual_view_first)


            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                if not self.rotated:
                    # Normalized xywh to pixel xyxy format
                    ### w, h reflect the resizing of the original image to one with max size no more than self.img_size (see load_image())
                    ### ratio reflects the distortion caused by aligning image size to 32x, (see letterbox())
                    ### we should avoid the case when ratio[0] != ratio[1], in which case the rotated bbox is not preserved ()
                    labels = x.copy()
                    labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                    labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                    labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                    labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
                else:
                    # Normalized xywhr to pixel xywhr format
                    labels = x.copy()
                    labels[:, 1] = ratio[0] * w * x[:, 1] + pad[0]  # pad width
                    labels[:, 2] = ratio[1] * h * x[:, 2] + pad[1]  # pad height
                    labels[:, 3] = ratio[0] * w * x[:, 3]
                    labels[:, 4] = ratio[1] * h * x[:, 4]
                    if self.tail:
                        labels[:, 6] = ratio[0] * w * x[:, 6]
                        labels[:, 7] = ratio[1] * h * x[:, 7]
                    

        if self.augment:
            # Augment imagespace
            if (not self.mosaic) and (not self.dual_view_first) and (not self.dual_view_second):
                if not self.rotated:
                    img, labels, invalid_mask = random_affine(img, labels,
                                                degrees=hyp['degrees'],
                                                translate=hyp['translate'],
                                                scale=hyp['scale'],
                                                shear=hyp['shear'], 
                                                mask=invalid_mask)
                else:
                    img, labels, invalid_mask = random_affine_xywhr(img, labels,
                                                degrees=hyp['degrees'],
                                                translate=hyp['translate'],
                                                scale=hyp['scale'], 
                                                mask=invalid_mask)

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            ### added to adapt to black-white image and color image at the same time
            if not (self.dual_view_first or self.dual_view_second):
                ### if dual_view, do this augmentation outside in dual view loader so that the dropped channels are the same for both views. 
                img, dropped_channels = augment_channel_drop(img, torch_mode=False)

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            if not self.rotated:
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width
            if self.tail:
                labels[:, 7] /= img.shape[0]  # height
                labels[:, 6] /= img.shape[1]  # width

        # if self.augment and not (self.dual_view_first or self.dual_view_second):   ### VISMODE1113 disable augment since it is not consistent between first view and second view: (No. It is oay to flip one of ori and bev. )
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if invalid_mask is not None:
                    invalid_mask = np.fliplr(invalid_mask)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
                    if self.rotated:
                        labels[:, 5] = - labels[:, 5]
                        if self.tail:
                            labels[:, 6] = - labels[:, 6]
                if self.dual_view_first:
                    bspec = bspec.flip(lr=True)
                elif self.dual_view_second:
                    calib = calib.flip(lr=True)

            ### TODO: rotated flag is not implemented here, since ud_flip is not used
            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if invalid_mask is not None:
                    invalid_mask = np.flipud(invalid_mask)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        if self.rotated:
            if self.tail:
                labels_out = torch.zeros((nL, 9))
            else:
                labels_out = torch.zeros((nL, 7))
        else:
            labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        if invalid_mask is not None:
            invalid_mask = invalid_mask[:, :, [0]].transpose(2, 0, 1)  # 1xH*W
            invalid_mask = np.ascontiguousarray(invalid_mask)
            invalid_mask = torch.from_numpy(invalid_mask)

        # print("VIS img.shape", img.shape, self.dual_view_first)

        if self.dual_view_first or self.dual_view_second:
            if self.dual_view_first:
                H_world_img = bspec.gen_H_world_bev()
                bspec_2 = bspec.scale(align_corners=False, scale_ratio_u=1/8, scale_ratio_v=1/8)
                bspec_4 = bspec_2.scale(align_corners=False, scale_ratio_u=0.5, scale_ratio_v=0.5)
                H_world_img_2 = bspec_2.gen_H_world_bev()
                H_world_img_4 = bspec_4.gen_H_world_bev()
                H_world_img_0 = bspec_0.gen_H_world_bev()
            elif self.dual_view_second:
                H_world_img = calib.gen_H_world_img()
                calib_2 = calib.scale(align_corners=False, scale_ratio_u=1/8, scale_ratio_v=1/8)
                calib_4 = calib_2.scale(align_corners=False, scale_ratio_u=0.5, scale_ratio_v=0.5)
                H_world_img_2 = calib_2.gen_H_world_img()
                H_world_img_4 = calib_4.gen_H_world_img()
                H_world_img_0 = calib_0.gen_H_world_img()
            
            H_world_img = torch.from_numpy(H_world_img)
            H_world_img_2 = torch.from_numpy(H_world_img_2)
            H_world_img_4 = torch.from_numpy(H_world_img_4)
            H_world_img_0 = torch.from_numpy(H_world_img_0)

            H_stack = torch.stack([H_world_img, H_world_img_2, H_world_img_4, H_world_img_0], dim=0)
            
            return torch.from_numpy(img), invalid_mask, labels_out, self.img_files[index], shapes, H_stack

        return torch.from_numpy(img), invalid_mask, labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, invalid_mask, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), invalid_mask, torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn_dual(batch):
        img, label, path, shapes, H_stacks = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), invalid_mask, torch.cat(label, 0), path, shapes, torch.stack(H_stacks, 0)

def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r < 1 or (self.augment and r != 1):  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

def load_image_mask(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    path = self.invalid_label_region_mask[index]
    img = cv2.imread(path)  # BGR
    if img is None or not self.use_mask:
        return None, None, None
    else:
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r < 1 or (self.augment and r != 1):  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])

def augment_strip_occlusion(img, torch_mode, img_blank=None, img_src=None):
    if torch_mode:
        C, H, W = img.shape
        img_np = np.ascontiguousarray(img.numpy().transpose(1,2,0))     ### must explicitly call ascontiguousarray, otherwise opencv does not work. 
    else:
        H, W, C = img.shape 
        img_np = img

    if img_blank is None:
        left = random.randrange(0, H)
        right = random.randrange(0, H)
        width = random.randint(5, 12)
        cv2.line(img_np, (0, left), (W-1, right), (0, 0, 0), width)

        img_blank = np.zeros_like(img_np)
        # img_blank = np.zeros(img_np.shape, np.uint8)
        gray_1 = random.randint(0, 255)
        gray_2 = random.randint(0, 255)
        gray_3 = random.randint(0, 255)
        black = random.randint(0, 9)
        if black == 0:
            gray_1 = 0
            gray_2 = 0
            gray_3 = 0
        cv2.line(img_blank, (0, left), (W-1, right), (gray_1, gray_2, gray_3), width)
    else:
        img_np[img_blank>0] = img_src[img_blank>0]

    if torch_mode:
        img = torch.from_numpy(img_np).permute(2, 0, 1)
    else:
        img = img_np

    return img, img_np, img_blank

def augment_channel_drop(img, torch_mode, dropout_chnl=None):

    if dropout_chnl is None:
        dropout_chnl = []
        if random.random() < 0.5:
            dropout_chnl.append(0)
        if random.random() < 0.5:
            dropout_chnl.append(1)
        if random.random() < 0.5:
            dropout_chnl.append(2)

        if len(dropout_chnl) == 3:
            dropout_chnl = []

    assert isinstance(dropout_chnl, list)

    if torch_mode:
        img = img.float()
        for chnl in dropout_chnl:
            img[chnl] = 0
    else:
        img = img.astype(np.float32)
        for chnl in dropout_chnl:
            img[..., chnl] = 0

    # n_left_chnl = 3 - len(dropout_chnl)
    # img = img / n_left_chnl * 3

    return img, dropout_chnl

    

def load_mosaic(self, index):
    # loads images in a mosaic
    ### TODO: rotated flag is not implemented here

    labels4 = []
    s = self.img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=-s // 2)  # border to remove

    return img4, labels4


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        # dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding   # should be 32? 20201224
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_affine_xywhr(img, targets=(), degrees=10, translate=.1, scale=.1, border=0, mask=None):
    ### to adapt to rotated bbox, we do not accept shear augmentation here. 
    ### different from random_affine() which takes xyxy as target input, 
    ### here we take xywhr as input (but denormalized to pixel) because we want to preserve the rectangle property

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)
    ### angle – Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner). [[c,s], [-s,c]]
    ### https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getrotationmatrix2d

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Combined rotation matrix
    M = T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        if mask is not None:
            mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        x0y0 = np.ones((n, 3))
        x0y0[:, :2] = targets[:, [1,2]]
        x0y0_new = (x0y0 @ M.T)[:, :2]

        wh = targets[:, [3,4]]
        wh_new = wh * s

        if targets.shape[1] == 6:
            yaw = targets[:, [5]]
            yaw_new = yaw + a*np.pi/180

            xywh_r = np.concatenate((x0y0_new, wh_new, yaw_new), axis=1)
        elif targets.shape[1] == 8:
            yaw = targets[:, [5]]
            yaw_new = yaw + a*np.pi/180

            x1y1 = np.ones((n, 3))
            x1y1[:, :2] = targets[:, [1,2]] + targets[:, [6,7]]
            x1y1_new = (x1y1 @ M.T)[:, :2]
            dxdy = x1y1_new - x0y0_new
            xywh_r = np.concatenate((x0y0_new, wh_new, yaw_new, dxdy), axis=1)
        else:
            xywh_r = np.concatenate((x0y0_new, wh_new), axis=1)

        x0 = x0y0_new[:,0]
        y0 = x0y0_new[:,1]
        w = wh_new[:,0]
        h = wh_new[:,1]
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio

        i = (w > 4) & (h > 4) & (x0 > 0) & (x0 < width-1) & (y0 > 0) & (y0 < height-1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:] = xywh_r[i]

    return img, targets, mask

def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0, mask=None):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        if mask is not None:
            mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))


    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets, mask


def cutout(image, labels):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def reduce_img_size(path='../data/sm4/images', img_size=1024):  # from utils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + '_reduced'  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob('%s/*.*' % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except:
            print('WARNING: image failure %s' % f)


def convert_images2bmp():  # from utils.datasets import *; convert_images2bmp()
    # Save images
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    # for path in ['../coco/images/val2014', '../coco/images/train2014']:
    for path in ['../data/sm4/images', '../data/sm4/background']:
        create_folder(path + 'bmp')
        for ext in formats:  # ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            for f in tqdm(glob.glob('%s/*%s' % (path, ext)), desc='Converting %s' % ext):
                cv2.imwrite(f.replace(ext.lower(), '.bmp').replace(path, path + 'bmp'), cv2.imread(f))

    # Save labels
    # for path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
    for file in ['../data/sm4/out_train.txt', '../data/sm4/out_test.txt']:
        with open(file, 'r') as f:
            lines = f.read()
            # lines = f.read().replace('2014/', '2014bmp/')  # coco
            lines = lines.replace('/images', '/imagesbmp')
            lines = lines.replace('/background', '/backgroundbmp')
        for ext in formats:
            lines = lines.replace(ext, '.bmp')
        with open(file.replace('.txt', 'bmp.txt'), 'w') as f:
            f.write(lines)


def recursive_dataset2bmp(dataset='../data/sm4_bmp'):  # from utils.datasets import *; recursive_dataset2bmp()
    # Converts dataset to bmp (for faster training)
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + '/' + file
            s = Path(file).suffix
            if s == '.txt':  # replace text
                with open(p, 'r') as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, '.bmp')
                with open(p, 'w') as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, '.bmp'), cv2.imread(p))
                if s != '.bmp':
                    os.system("rm '%s'" % p)


def imagelist2folder(path='data/coco_64img.txt'):  # from utils.datasets import *; imagelist2folder()
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)


def create_folder(path='./new_folder'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
