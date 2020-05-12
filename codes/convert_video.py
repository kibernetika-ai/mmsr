import logging
import time
import argparse
from collections import OrderedDict

import cv2
import numpy as np
import torch

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model


cfg = {
    "name": "EDVR",
    "suffix": None,
    "model": "video_base",
    "distortion": "sr",
    "scale": 4,
    "crop_border": None,
    "gpu_ids": [0],
    "datasets": {},
    "network_G": {
        "which_model_G": "EDVR",
        "nf": 128,
        "nframes": 5,
        "groups": 8,
        "front_RBs": 5,
        "back_RBs": 10,
        "predeblur": True,
        "HR_in": True,
        "w_TSA": False,
        "center": None,
    },
    "path": {
        "strict_load": False,
    },
    "is_train": False,
    "train": False,
    "dist": False,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output')

    return parser.parse_args()


def normalize(img: np.ndarray):
    img = img.astype(np.float32) / 255.0
    img = img.transpose([2, 0, 1])
    return torch.from_numpy(img).float()


def main():
    args = parse_args()
    cfg['path']['pretrain_model_G'] = args.model

    model = create_model(cfg)

    vc = cv2.VideoCapture(args.video)
    if args.output:
        fps = vc.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_format = video.get(cv2.CAP_PROP_FORMAT)
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(
            args.output, fourcc, fps,
            frameSize=(width, height)
        )

    log_frames = 100
    frame_num = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model.feed_data({'LQs': normalize(frame)}, need_GT=False)
        model.test()
        outputs = model.get_current_visuals(need_GT=False)
        __import__('ipdb').set_trace()
        output = outputs['rlt']

        frame_num += 1
        if frame_num % log_frames == 0:
            print(f'Processed {frame_num} frames.')

    vc.release()
    if args.output:
        video_writer.release()


if __name__ == '__main__':
    main()
