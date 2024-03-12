mkdir question_log

export CUDA_VISIBLE_DEVICES=1
nohup python question_generate.py  1 > question_log/1.log 2>&1&

export CUDA_VISIBLE_DEVICES=2
nohup python question_generate.py  2 > question_log/2.log 2>&1&

export CUDA_VISIBLE_DEVICES=3
nohup python question_generate.py  3 > question_log/3.log 2>&1&

export CUDA_VISIBLE_DEVICES=4
nohup python question_generate.py  4 > question_log/4.log 2>&1&

export CUDA_VISIBLE_DEVICES=5
nohup python question_generate.py  5 > question_log/5.log 2>&1&

