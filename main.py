from screen_inf import grab_screen_mss, grab_screen_win32, get_parameters
import cv2
import win32gui
import win32con
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--show-window', type=bool, default=True, help='Show real-time detection window.')
parser.add_argument('--region', type=tuple, default=(0.3, 0.5), help='Tracking range.(1.0, 1.0)means all screen, detect with the center of your screen.')
parser.add_argument('--resize-window', type=float, default=1, help='Change the real_time detection window size.')
args = parser.parse_args()


top_x, top_y, x, y = get_parameters()
len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))
monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}

while True:
    img0 = grab_screen_mss(monitor)
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