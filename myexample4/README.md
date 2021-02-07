# DeBERTa 模型测试
```
命令: 
python sequence_classfication.py --model_name_or_path microsoft/deberta-base --task_name sst2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir output/sst2/  --evaluation_strategy steps --eval_steps 500

epoch = 3.0
eval_accuracy = 0.9541284403669725
eval_loss = 0.21425390243530273
eval_runtime = 8.2135
eval_samples_per_second = 106.166

Electra base
python sequence_classfication.py --model_name_or_path google/electra-base-discriminator --task_name sst2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir output/sst2/  --evaluation_strategy steps --eval_steps 500

02/04/2021 05:22:46 - INFO - __main__ -   ***** Eval results sst2 *****
02/04/2021 05:22:46 - INFO - __main__ -     epoch = 3.0
02/04/2021 05:22:46 - INFO - __main__ -     eval_accuracy = 0.9472477064220184
02/04/2021 05:22:46 - INFO - __main__ -     eval_loss = 0.19900847971439362
02/04/2021 05:22:46 - INFO - __main__ -     eval_runtime = 7.2291
02/04/2021 05:22:46 - INFO - __main__ -     eval_samples_per_second = 120.623

#加上半精度
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


# 继续训练DeBERTa
python run_mlm.py  --model_name_or_path microsoft/deberta-base --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --output_dir output/mydeberta

python run_mlm.py \
    --model_name_or_path microsoft/deberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir output/mydeberta
    
# 分布式训练
## 节点1上运行:
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="192.168.50.139" --master_port=1234  run_mlm.py --model_name_or_path microsoft/deberta-base --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --output_dir output/mydeberta
## 节点2上运行:
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="192.168.50.139" --master_port=1234 run_mlm.py --model_name_or_path microsoft/deberta-base --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --output_dir output/mydeberta
