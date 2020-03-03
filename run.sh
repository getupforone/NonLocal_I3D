#!/bin/bash
rm -rf ./checkpoints
rm -rf ./runs
sleep 1
git pull
sleep 1
python run.py --cfg config/configs/kstartv/I3D_NLN_8x8_R50_KSTARTV.yaml 
