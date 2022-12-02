import torch
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import check_img_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'

imgsz = (640, 640)

# Load model
def load_model(args):
    device = ''
    device = select_device(device)
    model = DetectMultiBackend(args.weights, device=device, dnn=False, data='./data/coco128.yaml', fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size




