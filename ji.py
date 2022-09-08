import glob
import json
from PIL import Image
from utils.augmentations import letterbox
from pickle import TRUE
from utils.torch_utils import select_device
from utils.general import (check_img_size, non_max_suppression,
                           scale_coords, strip_optimizer)
import argparse
import numpy as np
import cv2
import os
import torch
# from icecream import ic as print
import torch.nn as nn

np.set_printoptions(suppress=True)

ROI_COLOR = {"unknown": 4, 'red': 2, 'blue': 0, "green": 1, "yellow": 3}

MODE = "test"
DEVICE = 0


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


colorsmap = np.array([[0, 0, 0],
                      [204, 102, 153],
                      [0, 255, 204],
                      [102, 102, 255],
                      [0, 0, 255],
                      [51, 102, 51],
                      [0, 255, 0],
                      [51, 153, 102],[51, 153, 102],[51, 153, 102]], dtype=np.uint8)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[512], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.25, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--line-thickness', default=3,
                        type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(FILE.stem, opt)
    return opt


def box_label(boxa, im, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = 3
    # (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    p1, p2 = list(boxa)

    # , thickness=lw, lineType=cv2.LINE_AA)
    cv2.rectangle(im, p1, p2, [128, 125, 50], lw, cv2.LINE_AA)
    # cv2.imshow("a", im)
    # cv2.waitKey(0)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3,
                               thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, [128, 125, 50], -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)

    return im


def xyxy2lxlywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


mode_papth = "/project/ev_sdk/src/runs/best.pt"
color_path = "/project/ev_sdk/src/runs/color_m.pt"


class MyMode():

    @torch.no_grad()
    def run(self, imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            line_thickness=3,  # bounding box thickness (pixels)
            ):

        img = letterbox(self.img0, imgsz,
                        stride=self.stride, auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)
        objs = []
        im0s = self.img0
        gain = min(im.shape[1] / im0s.shape[0],
                   im.shape[2] / im0s.shape[1])  # gain  = old / new
        pad = (im.shape[2] - im0s.shape[1] * gain) / \
            2, (im.shape[1] - im0s.shape[0] * gain) / 2  # wh padding

        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        # im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        predout = self.model(im, augment=augment)
        ####################### segmentation post process #######################
        segout = predout[1]
        segout = torch.nn.functional.softmax(segout, dim=1)
        segout = segout.squeeze(0)
        mask = torch.argmax(segout, dim=0)
        # print("segout",segout.shape, mask)
        mask = mask.detach().cpu().numpy()
        mask = mask.astype(np.uint8)
        oldshape = mask.shape
        mask = mask[int(pad[1]):int(oldshape[0] - pad[1]),
                    int(pad[0]):int(oldshape[1] - pad[0])]
        # np.savetxt("mask.txt",mask,fmt='%d')
        # pred_color = colorEncode(mask, colorsmap).astype(np.uint8)
        # print("out shape : ",type(predout),len(predout),predout[0][0].shape, predout[1].shape, im.shape)
        # NMS
        pred = non_max_suppression(
            predout[0][0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        col, plate, sp = None, [], None
        # Process predictions
        for i, det in enumerate(pred):  # per image
            # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im0s.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    xyxy_h = torch.tensor(xyxy).view(1, 4)
                    # normalized xywh
                    xywh = (xyxy2lxlywh(xyxy_h)).view(-1).tolist()
                    if c > 7:
                        plate.append([list(xyxy_h.numpy()[0].reshape(-1, 2).astype(np.int32)), self.names[8], 0.7, False])
                    else:
                        objs.append([*xywh, self.names[c], conf.tolist()])
                    # print()
                    xyxy_h_pt = xyxy_h.numpy()[0].reshape(-1, 2).astype(np.int32)
                    if c == 3:
                        col = self.get_ROI_Main_color(xyxy_h_pt, im0s)
                    # if MODE == "debug":
                        # im0 = im0s.copy()
                        # label = f'{self.names[c]} {conf:.2f}'
                        # box_label(xyxy_h_pt, im0, label, color=colorsmap[c])

                # maskcolor = cv2.resize(
                #     pred_color, (im0.shape[1], im0.shape[0]))
                # im_vis = cv2.addWeighted(im0, 0.6, maskcolor, 0.4, 1)
                # if MODE == "debug":
                # cv2.imwrite("res.png", im_vis)
                # cv2.imshow("a",im_vis)
                # cv2.waitKey()
        if False:
            # update model (to fix SourceChangeWarning)
            strip_optimizer(weights)
        # plate.append([[840, 250, 550, 750], self.names[8], 0.7, False])
        return cv2.resize(mask, (im0s.shape[1], im0s.shape[0])), objs, col, plate, sp

    def get_ROI_Main_color(self, pts, img0):
        x1, y1, x2, y2 = pts.reshape(-1)
        # print(x1, y1, x2, y2)

        img_roi = img0[y1:y2, x1:x2]
        img0 = cv2.resize(img_roi, (150, 150))
        img = torch.from_numpy(np.array([img0]).transpose((0, 3, 1, 2)))
        inputs = img.float().div(255).cuda()
        outputs = self.color_model(inputs)
        ret = (outputs.cpu()[0].numpy()).argmax()
        # print(ret)

        return self.color_map[ret]

    def __init__(self) -> None:

        self.device = select_device(DEVICE)

        if os.path.exists(mode_papth):
            ckpts = torch.load(mode_papth)
        else:
            ckpts = torch.load("/project/ev_sdk/src/runs/best.pt")

        if os.path.exists(color_path):
            checkpoint = torch.load(color_path)
        else:
            checkpoint = torch.load("/project/ev_sdk/src/runs/color.pt")
        # model = DetectMultiBackend(weights, device=device, dnn=dnn)
        self.model = ckpts['model'].float().cuda(DEVICE)

        self.stride, self.names = self.model.stride[-1].item(
        ), self.model.names
        self.opt = parse_opt()
        self.imgsz = check_img_size(
            self.opt.imgsz, s=self.stride)  # check image size
        dic = checkpoint["class_map"]
        print(dic)
        # ROI_COLOR  = {"unknown": 4, 'red': 2, 'blue': 0, "green": 1, "yellow": 3}
        self.color_map = dict(zip(dic.values(), dic.keys()))
        self.color_model = Net()
        self.color_model.load_state_dict(checkpoint["state_dict"])
        self.color_model = self.color_model.cuda(DEVICE)
        self.color_model.eval()

    def __call__(self, img):
        self.img0 = img
        pred_color, objs, col, plate, sp = self.run(**vars(self.opt))
        # print(objs)
        #x, y, width, height, name, score
        ans = {"object_detect": objs,
               "segment": pred_color,
               "color": col}
        return ans, plate


class Net(nn.Module):

    def __init__(self, class_num=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(6272, 512), nn.ReLU(),
            nn.Linear(512, class_num), nn.ReLU())
        # self.initialize_weights()

    def forward(self, x):
        return self.layers(x)

    def view(self):
        # print(model)
        x = torch.rand(size=(1, 3, 150, 150), dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
            print(layer.__class__.__name__, 'output:\t', x.shape)

    def initialize_weights(self):
        for m in self.layers():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def init():       # 模型初始化
    model = MyMode()  # "您的深度学习模型"  ###开发者需要自行设计深度学习模型
    return model


def check_plate_in_truck(truct, plate):
    tx, ty, tw, th, n, _ = truct
    pxyxy, n, cf, used = plate

    if used == True:
        return _, n, _, False
    px, py, px2, py2 = pxyxy
    if tx < px and ty < py and (tw + tx) > px2 and (th + ty) > py2:
        return pxyxy, n, cf, True
    else:
        return _, n, _, False


def process_image(net, input_image, args=None):
    results, plate = net(input_image)

    detect_objs = []
    for k, det in enumerate(results['object_detect']):
        x, y, width, height, name, score = det
        obj = {
            'name': name,
            'x': x,
            'y': y,
            'width': width,
            'height': height, 'confidence': float(score)
        }
        if name == 'open_bed_heavy_truck':
            '''
            开放式大型货车,需要识别车辆颜色，还有检测车牌的4个角点，开发者需要自行设计这些模型
            '''
            obj['plate'] = []
            obj['color'] = results["color"]

            for i, p in enumerate(plate):

                pt, cf, n, ret = check_plate_in_truck(det, p)
                if ret:
                    print(det, p, pt, cf)
                    plate[i][3]=True
                    obj['plate'].append({'name': n, 'points': pt, 'ocr': 'xxxxxxx',
                                         'confidence': cf})
        detect_objs.append(obj)

    mask = results['segment']
    args = json.loads(args)
    mask_output_path = args['mask_output_path']
    pred_mask_per_frame = Image.fromarray(mask)
    pred_mask_per_frame.save(mask_output_path)

    pred = {'model_data': {"objects": detect_objs, "mask": mask_output_path}}
    return json.dumps(pred)


if __name__ == "__main__":
    mode_papth = "runs/car.pt"
    MODE = "debug"
    color_path = "runs/color_m.pt"
    predictor = init()

    for i, path in enumerate(glob.glob("/home/data/1441/*.jpg")):
        original_image = cv2.imread(path)   # 读取图片
        args = {"mask_output_path": "mask.png"}
        result = process_image(predictor, original_image, json.dumps(args))
        # print(result)
        # with open('images/data.json', 'w', encoding='utf-8') as file:
        #     file.write(result)
