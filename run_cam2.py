#!/usr/bin/env python3
import argparse
import sys
import torch

import utils.checkpoint as cu
import utils.multiprocessing as mpu
from config.defaults import get_cfg

from test_net import test
from train_net import train
from cam_test2 import cam_test2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide NonLocalI3D img training and testing pipline. "
    )
    parser.add_argument(
        "--share_id",
        help="The shard id of current node, Starts from 0 to num_shareds - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def load_config(args):
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "num_shards") and hasattr(args,"shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args,"rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def main():
    #args = parse_args()
    #cfg  = load_config(args)
    cfg = get_cfg()
    cfg.merge_from_file('config/configs/kstartv/I3D_NLN_8x8_R50_KSTARTV2_10_12_1_16.yaml')
    cfg.TEST.BATCH_SIZE = 1
    cfg.NUM_GPUS = 1
    cfg.TEST.IS_CAM_TEST = True
    
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)

    if cfg.TEST.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    cam_test,
                    "tcp://localhost:9999",
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )

        else:
            print("single node used")
            cam_test2(cfg=cfg)

if __name__=="__main__":

    main()