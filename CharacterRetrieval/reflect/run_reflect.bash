mkdir reflect_log

export CUDA_VISIBLE_DEVICES=0
nohup python reflect.py --plot_number 0 > reflect_log/0.log 2>&1&

export CUDA_VISIBLE_DEVICES=1
nohup python reflect.py --plot_number 1 > reflect_log/1.log 2>&1&

export CUDA_VISIBLE_DEVICES=2
nohup python reflect.py --plot_number 2 > reflect_log/2.log 2>&1&

export CUDA_VISIBLE_DEVICES=3
nohup python reflect.py --plot_number 3 > reflect_log/3.log 2>&1&

export CUDA_VISIBLE_DEVICES=4
nohup python reflect.py --plot_number 4 > reflect_log/4.log 2>&1&

export CUDA_VISIBLE_DEVICES=5
nohup python reflect.py --plot_number 5 > reflect_log/5.log 2>&1&

