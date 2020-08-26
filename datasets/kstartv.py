
import os
import random
import torch
import torch.utils.data
import numpy as np
import utils.logging as logging
import datasets.sample as smpl
import datasets.transform as transform


logger = logging.get_logger(__name__)

class KstarTV(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        assert mode in [
            "train","val","test"
        ], "split '{}' not supported for kstartv".format(mode)
        self.mode = mode
        self.cfg = cfg

        # For training or validation mode, one signle clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NIM_SPATIAL_CROPS is cropped spatially from
        # the frames
        self._img_meta = {}
        
        self._num_clips = 1
        
        # if self.mode in ["train", "val"]:
        #     self._num_clips = 1
        # elif self.mode in ["test"]:
        #     self._num_clips = (
        #         cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        #     )
        logger.info("Constructing kstartv {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.txt".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_seq_imgs  = []
        self._labels = []
        self._spatial_temporal_idx = []

        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                if len(path_label.split()) == 2 :
                    path, label = path_label.split()
                    for idx in range(self._num_clips):
                        self._path_to_seq_imgs.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                        )
                        intlabel = -1
                        if label == 'True':
                            intlabel = 1
                        elif label  == 'False':
                            intlabel = 0
                        self._labels.append(int(intlabel))
                        self._spatial_temporal_idx.append(idx)
                        self._img_meta[clip_idx * self._num_clips + idx] = {}
                elif len(path_label.split()) == 1 :
                        self._path_to_seq_imgs.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                        )
                        self._labels.append(int(-1))
                        self._spatial_temporal_idx.append(idx)
                        self._img_meta[clip_idx * self._num_clips + idx] = {}
        assert(
            len(self._path_to_seq_imgs) > 0
        ), "Failed to load davis split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing davis dataloader (size: {}) from {}".format(
                len(self._path_to_seq_imgs), path_to_file
            )
        )
    def pack_pathway_output(self, frames):
       frame_list = [frames]
       return frame_list

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,#min_scale=256,
        max_scale=320,#max_scale=320,
        crop_size=224,#crop_size=224,
    ):
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = transform.random_crop(frames, crop_size)
            frames = transform.horizontal_flip(0.5, frames)
        else:
            assert len({min_scale, max_scale, crop_size}) == 1
            frames = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = transform.uniform_crop(frames, crop_size,spatial_idx)
        return frames

    def __len__(self):
        return len(self._path_to_seq_imgs)

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and images
        index. 
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the images. The dimension
                is `channel` x `num_frames` x `height` x `width`.
            label (int): the label of the current images.
            index (int): if the images provided by pytorch sampler can be 
                read, then return the index of the images. 
        """
        # if self.mode in ["train", "val"]:
        #     temporal_sample_index = -1
        #     spatial_sample_index = -1
        #     min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
        #     max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
        #     crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        # elif self.mode in ["test"]:
        #     temporal_sample_index = (
        #         self._spatial_temporal_idx[index]
        #         // self.cfg.TEST.NUM_SPATIAL_CROPS
        #     )
            
        #     # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
        #     # center, or right if width is larger than height, and top, middle,
        #     # or bottom if height is larger than width
            
        #     spatial_sample_index = (
        #         self._spatial_temporal_idx[index]
        #         % self.cfg.TEST.NUM_SPATIAL_CROPS
        #     )

        #     # The testing is deterministic and no jitter should be performed.
        #     # min_scale, max_scale, and crop_size are expect to be the same

        #     min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        #     assert len({min_scale, max_scale, crop_size}) == 1
        if self.mode in ["train", "val","test"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)    
            )
        path_to_frames = []
        try:            
            seq_dir = self._path_to_seq_imgs[index].strip()
            name_of_frames = np.sort(os.listdir(seq_dir))
            #print("davis:__getitem__:seq_dir: {}".format(seq_dir))
            
            for name in name_of_frames:
                path_to_frame = os.path.join(seq_dir,name)
                path_to_frames.append(path_to_frame)   
            #    print("davis:__getitem__:name: {}".format(name))   
            #path_to_frames_length = len(path_to_frames)           
        except Exception as e:
            logger.info(
                "Failed to load video from {} with error {}".format(
                    self._path_to_seq_imgs[index],e
                )
            )
        frames = smpl.get_frames(
            path_to_frames,
            self.cfg.DATA.SAMPLING_RATE, # sampling_rate 1 # The video sampling rate of the input clip.
            self.cfg.DATA.NUM_FRAMES,       #num_frames 10 # The number of frames of the input clip.
            temporal_sample_index,          #clip_idx
            self.cfg.TEST.NUM_ENSEMBLE_VIEWS,   #num_clips = 10 #video_meta=self._video_meta[index],     
            fps=1,
            target_fps=1,                  #target_fps
            rz_size= crop_size,
            )
        # if frames is None:
        #     index = random.randint(0, len(self._path_to_videos) - 1)
        #     continue
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)
        # T H W C -> C T H W
        frames = frames.permute(3, 0, 1, 2)

        # frames = self.spatial_sampling(
        #     frames,
        #     spatial_idx=spatial_sample_index,
        #     min_scale=min_scale,
        #     max_scale=max_scale,
        #     crop_size=crop_size,
        # )

        #print("after spatial_sampling frames len ", frames.shape)
        #print("after spatial_sampling frames len = {}".format(frames.shape[1]))
        label = self._labels[index]
        #frames = self.pack_pathway_output(frames) # this parts makes tensor to list of tensor
        return frames, label, index
 

            #gt = cv2.imread(gt_path, -1) # load unchaned image