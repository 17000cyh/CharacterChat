mkdir question2_log

export CUDA_VISIBLE_DEVICES=1
nohup python question_generate.py  6 > question2_log/1.log 2>&1&

export CUDA_VISIBLE_DEVICES=2
nohup python question_generate.py  7 > question2_log/2.log 2>&1&

export CUDA_VISIBLE_DEVICES=3
nohup python question_generate.py  8 > question2_log/3.log 2>&1&

export CUDA_VISIBLE_DEVICES=4
nohup python question_generate.py  9 > question2_log/4.log 2>&1&

export CUDA_VISIBLE_DEVICES=5
nohup python question_generate.py  10 > question2_log/5.log 2>&1&

