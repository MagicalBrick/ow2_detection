import cv2
import numpy as np
import win32gui
import win32con
import argparse
import torch
from utils.torch_utils import select_device
from utils.general import check_img_size, Profile, non_max_suppression
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from screen_inf import grab_screen_mss, get_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--show-window', type=bool, default=True, help='Show real-time detection window.')
parser.add_argument('--weights', nargs='+', type=str, default='./ow2_s.pt', help='model path or triton URL')
parser.add_argument('--region', type=tuple, default=(0.3, 0.5), help='Tracking range.(1.0, 1.0)means all screen and detect with the center of your screen.')
parser.add_argument('--resize-window', type=float, default=1, help='Change the real_time detection window size.')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
args = parser.parse_args()

# set the monitor info
top_x, top_y, x, y = get_parameters()
len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))
monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}

# Load model
device = ''
device = select_device(device)
model = DetectMultiBackend(args.weights, device=device, dnn=False, data='./data/coco128.yaml', fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(args.imgsz, s=stride)  # check image size

source = str('screen')
screenshot = source.lower().startswith('screen')

while True:
    img0 = grab_screen_mss(monitor)
    img = cv2.resize(img0, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = letterbox(img, imgsz, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = np.reshape(img, (1, 3, 640, 640))

    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # preprocessing
    with dt[0]:
        im = torch.from_numpy(img).to(model.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        visualize = False
        pred = model(im, augment=False, visualize=visualize)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

    if args.show_window:
        cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('detection', int(len_x * args.resize_window), int(len_y * args.resize_window))
        cv2.imshow('detection', img0)

    hwnd = win32gui.FindWindow(None, 'detection')
    CVRECT = cv2.getWindowImageRect('detection')
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.destoryAllWindows()
        break
