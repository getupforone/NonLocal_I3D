#!/bin/bash
rm -rf ./checkpoints
sleep 1
git pull
sleep 1
python run.py --cfg config/configs/davis/I3D_NLN_8x8_R50_KSTAR.yaml 
