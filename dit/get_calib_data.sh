CFG='w4a4.yaml'
EXP_NAME="get_calib_data"
python get_calib_data.py --ptq-config "./configs/${CFG}" --log "./logs/${EXP_NAME}" --cfg-scale 1.5 --num-sampling-steps 100