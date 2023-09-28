DATA_DIR=./data
SIZE=base # lets use large, (slower than base, but still quite fast and accessible, but less accurate than xl or xxl)

# # download the NQ data
# python preprocessing/prepare_qa.py --output_directory ${DATA_DIR}/data/
# # download the Wikipedia 2018 corpus
# python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2018 --output_directory ${DATA_DIR} 
# # downloads pretrained Atlas-large
# python preprocessing/download_model.py --model models/atlas/${SIZE} --output_directory ${DATA_DIR}  

TRAIN_FILE="${DATA_DIR}/data/nq_data/train.64-shot.jsonl"
EVAL_FILES="${DATA_DIR}/data/nq_data/dev.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=my-nq-64-shot-example
TRAIN_STEPS=60
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=29500 train.py \
    --shuffle \
    --train_retriever \
    --gold_score_mode pdist \
    --use_gradient_checkpoint_reader --use_gradient_checkpoint_retriever\
    --precision fp32 \
    --shard_optim --shard_grads \
    --temperature_gold 0.01 --temperature_score 0.01 \
    --refresh_index -1 \
    --query_side_retriever_training\
    --target_maxlength 16 \
    --reader_model_type llama2 \
    --dropout 0.1 --weight_decay 0.01 --lr 4e-5 --lr_retriever 4e-5 --scheduler linear \
    --text_maxlength 512 \
    --model_path "${DATA_DIR}/models/atlas/${SIZE}"\
    --reader_model_path /mnt/nvme_workspace1/llm_data/llm_model/Llama-2-7b-chat-hf\
    --train_data "${DATA_DIR}/data/nq_data/train.64-shot.jsonl" \
    --eval_data "${DATA_DIR}/data/nq_data/dev.jsonl" \
    --per_gpu_batch_size 1 \
    --n_context 20 \
    --retriever_n_context 20 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq ${TRAIN_STEPS} \
    --log_freq 4 \
    --total_steps ${TRAIN_STEPS} \
    --warmup_steps 5 \
    --save_freq ${TRAIN_STEPS} \
    --write_results \
    --task vanilla_qa \
    --index_mode flat \
    --passages "${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl" "${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl" \
    --load_index_path '/mnt/nvme_workspace1/llm_data/wiki/large'