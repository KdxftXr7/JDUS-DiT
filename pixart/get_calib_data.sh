LOG='calib_data'
CFG='w4a4.yaml'
PROMPT_PATH='assets/samples_8.txt'
python get_calib_data.py --quant-config "./configs/${CFG}" --log "./logs/${LOG}"  --prompt $PROMPT_PATH