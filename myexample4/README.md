# DeBERTa 模型测试
```
命令: 
python sequence_classfication.py --model_name_or_path microsoft/deberta-base --task_name sst2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir output/sst2/

python sequence_classfication.py \
  --model_name_or_path microsoft/deberta-base \
  --task_name sst2 \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir output/sst2/ \
  --fp16
```