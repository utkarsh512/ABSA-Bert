# Shell script for distributed training of a model on GPU
# How to use:
# If you want to train 'QA_M' model with maximum allowed sequence length 100 and batch size of 32 with amp (if without amp, make last argument 0), then write
#
# $ bash train.sh 100 32 1
#
# @author utkarsh512

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch run_classifier_TABSA.py \
--task_name sentihood_${1} \
--data_dir data/sentihood/bert-pair/ \
--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length ${2} \
--train_batch_size ${3} \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--output_dir result/sentihood/${1} \
--seed 42 \
--amp ${4}
