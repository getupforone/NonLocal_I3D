import math
import numpy as np
import random
import torch
import cv2

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames

def get_start_end_idx(seq_size, clip_size, clip_idx, num_clips):
    delta = max(seq_size - clip_size, 0)
    if clip_idx == -1:
        start_idx = random.uniform(0, delta)
    else:
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx
    
def rd_frames(path_to_frames):
    frames = []
    try:
        for path in path_to_frames:
            frame = cv2.imread(path, -1) # load unchaned image
            frames.append(frame)
            #print(path)

            #height, width, channels = frame.shape
            #print(path)
            #print("sizeof img = {}/{}/{}".format(height, width,channels))
            
    except Exception as e:
        print("Failed to get images with exception: {}",format(e))
        return None
    return frames

def cvt_frames(frames):
    cv2.imshow('frame',frames[0])
    cv2.waitKey(0)
    return torch.as_tensor(np.stack(frames))

def get_frames(path_to_frames,
                sampling_rate,
                num_frames,
                clip_idx=-1,
                num_clips=10,
                fps=120,
                target_fps=30,
):
    assert clip_idx >= -1, "Not valid clip_idx {}".format(clip_idx)
    frames = None
    frames = rd_frames(path_to_frames)
    #print("frames types is {}".format(type(frames)))
    #print("frames len is {}".format(len(frames)))
    frames = cvt_frames(frames)

    start_idx, end_idx = get_start_end_idx(
        frames.shape[0],
        num_frames * sampling_rate * fps / target_fps,
        clip_idx,
        num_clips,
    )

    frames = temporal_sampling(frames,start_idx,end_idx,num_frames)
    return frames

