"""
Modified from https://github.com/m-bain/frozen-in-time/blob/22a91d78405ec6032fdf521ae1ff5573358e632f/base/base_dataset.py
"""
import random
import io
import av
import cv2
import decord
import imageio
from decord import VideoReader
import torch
import numpy as np
import math
decord.bridge.set_bridge("torch")

import logging
logger = logging.getLogger(__name__)

def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    )
    return float(video_duration)


def get_frame_indices_by_fps():
    pass


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_av(video_path, num_frames, sample='rand', fix_start=None, max_num_frames=-1):
    reader = av.open(video_path)
    frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    vlen = len(frames)
    duration = get_pyav_video_duration(reader)
    fps = vlen / float(duration)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = torch.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, duration


def read_frames_gif(
        video_path, num_frames, sample='rand', fix_start=None, 
         max_num_frames=-1, client=None, trimmed30=False,
    ):
    gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        max_num_frames=max_num_frames
    )
    frames = []
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    frames = torch.stack(frames)  # .float() / 255
    return frames, frame_indices, None

def read_frames_cv2(
    video_path, num_frames, sample='rand', fix_start=None,
    max_num_frames=-1, client=None, trimmed30=False
    ):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    if 2 <= duration <= 30:
        # Extract all frames for videos between 2 and 30 seconds
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames
    elif 30 < duration < 60:
        # Extract frames from the middle 30 seconds
        start_frame = int((total_frames - 30 * fps) / 2)
        end_frame = start_frame + int(30 * fps)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames
    else:
        cap.release()
        logger.warning(f"Video duration is not between 2 and 60 seconds: {duration}")
        return None


def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, trimmed30=False
    ):
    from ....video_utils import get_video_details, load_frames_from_video
    import numpy as np
    total_frames, original_fps, video_duration = get_video_details(video_path)
    uniform_sampling = min(num_frames, total_frames)
    all_indices = np.linspace(0, total_frames - 1, uniform_sampling, dtype=int)
    frames = load_frames_from_video(video_path, all_indices, "decord", True)
    frames = frames.permute(0, 3, 1, 2) # (T, C, H, W), torch.uint8
    return frames, all_indices, video_duration # New code written by ZQ, verified to be the same except this take 0 frame but the original code skip 0 frame

    # frames_by_cv2 = read_frames_cv2(
    #     video_path, num_frames, sample, fix_start, max_num_frames, client, trimmed30
    # ) # ZQ: will return same results
    # if video_path.startswith('s3') or video_path.startswith('p2'):
    #     video_bytes = client.get(video_path)
    #     video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    # else:
    #     video_reader = VideoReader(video_path, num_threads=1)
    # vlen = len(video_reader)
    # fps = video_reader.get_avg_fps()
    # duration = vlen / float(fps)

    # # only use top 30 seconds
    # if trimmed30 and duration > 30:
    #     duration = 30
    #     vlen = int(30 * float(fps))

    # frame_indices = get_frame_indices(
    #     num_frames, vlen, sample=sample, fix_start=fix_start,
    #     input_fps=fps, max_num_frames=max_num_frames
    # )
    # frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    # import pdb; pdb.set_trace()
    # frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    # return frames, frame_indices, duration


VIDEO_READER_FUNCS = {
    'av': read_frames_av,
    'decord': read_frames_decord,
    'gif': read_frames_gif,
}
