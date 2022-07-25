#!/bin/bash
export TRAIN_PATH="data/SF_ATIS/train.json"
export DEV_PATH="data/SF_ATIS/dev.json"
export TEST_PATH="data/SF_ATIS/test.json"
export CHAR_VOCAB_PATH="data/charindex.json"
export INTENT_LABEL_PATH="data/SF_ATIS/intent_label.txt"
export SLOT_LABEL_PATH="data/SF_ATIS/slot_label.txt"
export MAX_CHAR_LEN=20
export MAX_SEQ_LENGTH=100
export BATCH_SIZE=32
export CHAR_EMBEDDING_DIM=120
export CHAR_HIDDEN_DIM=150
export NUM_BERT_LAYER=1
export CHAR_VOCAB_SIZE=108
export HIDDEN_DIM=728
export HIDDEN_DIM_FFW=300
export NUM_SLOT_LABELS=83
export NUM_INTENT_LABELS=25
export MODEL_NAME_OR_PATH="vinai/phobert-base"
export NUM_EPOCHS=50
export LEARNING_RATE=5e-5
export ADAM_EPSILON=1e-8
export WEIGHT_DECAY=0.01
export WARMUP_STEPS=0
export MAX_GRAD_NORM=1
export SAVE_FOLDER="results"

python3 main.py --train_path $TRAIN_PATH \
                --max_char_len $MAX_CHAR_LEN  \
                --dev_path $DEV_PATH \
                --max_seq_length $MAX_SEQ_LENGTH \
                --batch_size $BATCH_SIZE \
                --char_embedding_dim $CHAR_EMBEDDING_DIM \
                --char_hidden_dim $CHAR_HIDDEN_DIM \
                --num_layer_bert $NUM_BERT_LAYER \
                --char_vocab_size $CHAR_VOCAB_SIZE \
                --hidden_dim $HIDDEN_DIM \
                --hidden_dim_ffw $HIDDEN_DIM_FFW \
                --num_slot_labels $NUM_SLOT_LABELS \
                --num_intent_labels $NUM_INTENT_LABELS\
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --num_epochs $NUM_EPOCHS \
                --learning_rate $LEARNING_RATE \
                --adam_epsilon $ADAM_EPSILON \
                --weight_decay $WEIGHT_DECAY \
                --warmup_steps $WARMUP_STEPS \
                --max_grad_norm $MAX_GRAD_NORM  \
                --save_folder $SAVE_FOLDER \
                --test_path $TEST_PATH \
                --char_vocab_path $CHAR_VOCAB_PATH \
                --intent_label_path $INTENT_LABEL_PATH \
                --slot_label_path $SLOT_LABEL_PATH\
                --do_eval \
                --do_train \
                --use_char \
                --use_crf