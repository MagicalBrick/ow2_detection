import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'

weight = './best_ow2_s.pt'
imgsz = 640
