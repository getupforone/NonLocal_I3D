import os
import random
import torch
import torch.utils.data
import numpy as np
import utils.logging as logging
import datasets.sample as smpl
import datasets.transform as transform


logger = logging.get_logger(__name__)

class Davis(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        assert mode in [
            "train","val","test"
        ], "split '{}' not supported for Davis".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._img_meta = {}
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )
        logger.info("Constructing Davis {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}_list.txt".format(self.mode)
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
                        self._labels.append(int(label))
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
        min_scale=64,#min_scale=256,
        max_scale=80,#max_scale=320,
        crop_size=56,#crop_size=224,
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
        if self.mode in ["train", "val"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)    
            )
        try:            
            seq_dir = self._path_to_seq_imgs[index].strip()
            path_to_frames = np.sort(os.listdir(seq_dir))
            path_to_frames_length = len(path_to_frames)
            print("!!! length={}".path_to_frames_length)
        except Exception as e:
            logger.info(
                "Failed to load video from {} with error {}".format(
                    self._path_to_seq_imgs[index],e
                )
            )
        frames = smpl.get_frames(
            path_to_frames,
            self.cfg.DATA.SAMPLING_RATE,
            self.cfg.DATA.NUM_FRAMES,
            temporal_sample_index,
            self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
            #video_meta=self._video_meta[index],
            target_fps=30,
            )
        # if frames is None:
        #     index = random.randint(0, len(self._path_to_videos) - 1)
        #     continue
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.DATA.STD)
        # T H W C -> C T H W
        frames = frames.permute(3, 0, 1, 2)

        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        label = self._labels[index]
        frames = self.pack_pathway_output(frames)
        return frames, label, index
 

            #gt = cv2.imread(gt_path, -1) # load unchaned image



