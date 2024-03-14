mkdir reflect2_log


export CUDA_VISIBLE_DEVICES=1
nohup python reflect.py --plot_number 7 > reflect2_log/1.log 2>&1&

export CUDA_VISIBLE_DEVICES=2
nohup python reflect.py --plot_number 8 > reflect2_log/2.log 2>&1&

export CUDA_VISIBLE_DEVICES=3
nohup python reflect.py --plot_number 9 > reflect2_log/3.log 2>&1&

export CUDA_VISIBLE_DEVICES=4
nohup python reflect.py --plot_number 10 > reflect2_log/4.log 2>&1&

export CUDA_VISIBLE_DEVICES=5
nohup python reflect.py --plot_number 11 > reflect2_log/5.log 2>&1&

