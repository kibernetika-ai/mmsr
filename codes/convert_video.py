import logging
import time
import argparse

import cv2
import numpy as np
import torch

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
        "nframes": 3,
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
    parser.add_argument('--show', action='store_true')

    return parser.parse_args()


def normalize(img: np.ndarray):
    rank = len(img.shape)
    height_dim = 1 if rank == 4 else 0
    nearest_multiple_16 = img.shape[height_dim] // 16 * 16
    if nearest_multiple_16 != img.shape[height_dim]:
        # crop by height
        crop_need = img.shape[height_dim] - nearest_multiple_16
        if rank == 4:
            img = img[:, crop_need // 2:-crop_need // 2, :, :]
        else:
            img = img[crop_need // 2:-crop_need // 2, :, :]

    img = img.astype(np.float32) / 255.0
    img = img.transpose([2, 0, 1])
    return torch.from_numpy(img).float()


def denormalize(tensor):
    numpy = tensor.detach().cpu().numpy()
    img = (numpy * 255.0).clip(0, 255).astype(np.uint8)
    return img.transpose([1, 2, 0])


def main():
    args = parse_args()
    cfg['path']['pretrain_model_G'] = args.model

    print(f'Loading model from {args.model}...')
    model = create_model(cfg)
    print(f'Done.')

    vc = cv2.VideoCapture(args.video)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = 3
    if args.output:
        fps = vc.get(cv2.CAP_PROP_FPS) / n_frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_format = video.get(cv2.CAP_PROP_FORMAT)
        video_writer = cv2.VideoWriter(
            args.output, fourcc, fps,
            frameSize=(width, height // 16 * 16)
        )

    log_frames = 100
    frame_num = 0
    frame_processed = 0
    time_sum = 0
    imgs_in = []
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        frame_num += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = normalize(frame)
        if len(imgs_in) <= n_frames:
            imgs_in.append(img)
            if len(imgs_in) < n_frames:
                continue
            if len(imgs_in) > n_frames:
                imgs_in = [imgs_in[-1]]
                continue

        data = {'LQs': torch.from_numpy(np.expand_dims(np.stack(imgs_in), axis=0))}
        model.feed_data(data, need_GT=False)

        t = time.time()
        model.test()
        time_sum += time.time() - t

        outputs = model.get_current_visuals(need_GT=False)
        output = outputs['rlt']
        output_frame = denormalize(output)

        frame_processed += 1
        if frame_processed % log_frames == 0:
            print(f'Processed {frame_processed} frames.')

        cv_frame = output_frame[:, :, ::-1]
        if args.output:
            video_writer.write(cv_frame)
        if args.show:
            cv2.imshow('Video', cv_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

    print(f'Average inference time: {time_sum / frame_processed * 1000:0.3f}ms')
    vc.release()
    if args.output:
        video_writer.release()


if __name__ == '__main__':
    main()
