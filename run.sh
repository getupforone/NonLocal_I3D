#!/bin/bash
rm -rf ./checkpoints
rm -rf ./runs
sleep 1
#git pull
sleep 1
#python run.py --cfg config/configs/kstartv/I3D_NLN_8x8_R50_KSTARTV3_12_12_1_32.yaml 
python run.py --cfg config/configs/kstartv/I3D_NLN_8x8_R50_KSTARTV2_12_8_1_32.yaml 