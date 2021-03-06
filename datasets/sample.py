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

def temporal_sampling_select(seq_size, start_idx, end_idx, num_samples):
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, seq_size - 1).long()
    return index

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
        print("Failed to get images with exception: {}".format(e))
        return None
    return frames

def rd_frames_select(path_to_frames, index,rz_size):
    frames = []
    #path_to_frames_select = torch.index_select(path_to_frames, 0, index)
    path_to_frames_np = np.array(path_to_frames)
    path_to_frames_select = path_to_frames_np[index]

    try:
        for path in path_to_frames_select:
            frame = cv2.imread(path, -1) # load unchaned image
            if rz_size !=-1:
                rz_frame = resize_frame(frame,rz_size)
                frames.append(rz_frame)
            else:
                frames.append(frame)
            #print(path)

            #height, width, channels = frame.shape
            #print(path)
            #print("sizeof img = {}/{}/{}".format(height, width,channels))
            
    except Exception as e:
        print("Failed to get images with exception: {}".format(e))
        return None
    return frames

def resize_frame(frame, rz_size):
    height, width, channels = frame.shape
    if height == rz_size and width== rz_size:
        return frame
    rz_img = cv2.resize(frame,(rz_size,rz_size),interpolation = cv2.INTER_CUBIC)
    return rz_img

def cvt_frames(frames):
    #cv2.imshow('frame',frames[0])
    #cv2.waitKey(0)
    # print("cvt_frames: fames shape {}".format(len(frames)))
    # for idx, elm in enumerate(frames):
    #     print("\n frames[{}]: {}".format(idx, elm.shape))
    frames = torch.as_tensor(np.stack(frames))
    return frames


def get_start_end_idx(seq_size, clip_size, clip_idx, num_clips):
    delta = max(seq_size - clip_size, 0)
    if clip_idx == -1:
        start_idx = random.uniform(0, delta)
    else:
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx

def get_frames(path_to_frames,
                sampling_rate,  # 1 frame sampling rate( interval between two sampled frames)
                num_frames,     # 8
                clip_idx=-1,
                num_clips=10,
                fps=1,
                target_fps=1,
                rz_size=244,
):
    assert clip_idx >= -1, "Not valid clip_idx {}".format(clip_idx)
    frames = None
    #frames = rd_frames(path_to_frames)
    
    #print("frames types is {}".format(type(frames)))
    #print("frames len is {}".format(len(frames)))
    #print("path_to_frames is {}".format(len(path_to_frames)))
    seq_size = len(path_to_frames)
    #frames = cvt_frames(frames)
    #print("frames types is {}".format(type(frames)))
    #print("num_of_frames1 {}".format(frames.shape))
    start_idx, end_idx = get_start_end_idx(
        seq_size, #frames.shape[0],                                    #seq_size
        num_frames * sampling_rate * target_fps / fps,      #clip_size
        clip_idx,                                           #clip_idx
        num_clips,                                          #num_clips
    )

    #frames = temporal_sampling(frames,start_idx,end_idx,num_frames)
    index = temporal_sampling_select(seq_size,start_idx,end_idx,num_frames)
    frames = rd_frames_select(path_to_frames, index,rz_size)
    frames = cvt_frames(frames)
    #print("num_of_frames2 {}".format(frames.shape))
    return frames

